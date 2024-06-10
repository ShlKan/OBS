#include "Dialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <mlir/IR/Attributes.h>

using namespace mlir;
using namespace obs;

namespace {

struct ReadOptimizePass
    : public mlir::PassWrapper<ReadOptimizePass, OperationPass<obs::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReadOptimizePass)

  void runOnOperation() override {
    auto f = getOperation();
    llvm::SmallVector<mlir::Operation *, 16> AllOperations;
    f->walk([&](mlir::Operation *op) { AllOperations.push_back(op); });

    if (AllOperations.empty())
      return;

    for (auto *op = AllOperations.begin(); op != AllOperations.end() - 1;
         op++) {
      if (llvm::isa<ReadOp>(*op) && llvm::isa<ReadOp>(*(op + 1))) {
        (*op)->erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::obs::createReadOptimizePass() {
  return std::make_unique<ReadOptimizePass>();
}