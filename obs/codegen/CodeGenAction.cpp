

#include "CodeGenAction.h"
#include "CodeGen.h"
#include "Dialect.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "Passes.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclGroup.h>

#include <iostream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <ostream>

using namespace clang;

namespace mlir {
namespace obs {

void CodeGenConsumer::HandleTranslationUnit(ASTContext &context) {
  MLIRContext codegenContext;
  codegenContext.getOrLoadDialect<OBSDialect>();
  auto Module = mlirGen(codegenContext, context);

  if (EnableOpt) {
    mlir::registerPassManagerCLOptions();

    mlir::PassManager pm(Module.get()->getName());

    if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
      exit(4);

    mlir::OpPassManager &optPM = pm.nest<mlir::obs::FuncOp>();
    optPM.addPass(mlir::obs::createReadOptimizePass());

    if (mlir::failed(pm.run(*Module)))
      exit(4);
  }

  Module->dump();
}

} // namespace obs
} // namespace mlir