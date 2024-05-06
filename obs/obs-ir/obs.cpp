#include "AST.h"
#include "Lexer.h"
#include "Parser.h"

#include "Dialect.h"
#include "MLIRGen.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>
#include <memory>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <string>
#include <system_error>


using namespace obs;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional, 
                                          cl::desc("<input obs file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
  enum InputType { OBS, MLIR };
} //namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(OBS), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(OBS, "obs", "load the input file as a OBS source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST, DumpMLIR, DumpMLIRAffine }; 
} // namespace

static cl::opt<enum Action> emitAction("emit", cl::desc("Select the kind of output desired"), 
                           cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
                           cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
                           cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine", 
                                      "output the MLIR dump after affine lowering")));
                          
static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

std::unique_ptr<obs::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get() ->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.toy' input to the compiler.                                                                                                                             
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.                                                                                                                                 
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.                                                                                                                                            
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int dumpMLIR() {

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::obs::OBSDialect>();
  
  mlir::OwningOpRef<mlir::ModuleOp> module ;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  if (int error = loadMLIR(sourceMgr, context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;

  if (enableOpt || isLoweringToAffine) {
    pm.addPass(mlir::createInlinerPass());


    mlir::OpPassManager &optPM = pm.nest<mlir::obs::FuncOp>();
    // Add a run of the canonicalizer to optimize the mlir module. 
    optPM.addPass(mlir::obs::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());                                                                                                   

    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToAffine) {
    pm.addPass(mlir::obs::createLowerToAffinePass());

    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    if (enableOpt) {
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }

  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  module->dump();
  return 0; 
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a OBS AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.                                                                                                                               
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "obs compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
  case Action::DumpMLIRAffine:
    return dumpMLIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;

}

