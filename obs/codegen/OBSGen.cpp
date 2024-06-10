
#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <string>
#include <vector>

#include "OBSGen.h"
using namespace llvm;

static cl::opt<std::string>
    inputFileName(cl::Positional, cl::desc("<input file name>"), cl::init("-"));

static cl::opt<std::string> doubleDash(cl::Positional, "",
                                       cl::desc("Accept --"));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

static cl::opt<std::string> cPlusPlus("std", cl::desc("c++ standard"));

// `main` function translates a C program into a OBS.
int main(int argc, const char **argv) {

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  if (enableOpt) {
    std::cout << "Optimization is enabled" << std::endl;
  }

  llvm::cl::OptionCategory CodeGenCategory("OBS code generation");
  auto OptionsParser =
      clang::tooling::CommonOptionsParser::create(argc, argv, CodeGenCategory);

  if (!OptionsParser) {
    // Fail gracefully for unsupported options.
    llvm::errs() << OptionsParser.takeError() << "error";
    return 1;
  }

  auto sources = OptionsParser->getSourcePathList();

  clang::tooling::ClangTool Tool(OptionsParser->getCompilations(), sources);

  Tool.run(new mlir::obs::CodeGenFrontendActionFactory(enableOpt));
}
