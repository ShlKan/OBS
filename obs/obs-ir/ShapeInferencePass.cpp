#include "ShapeInferenceInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "Dialect.h"
#include "Passes.h"
#include "ShapeInferenceInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace obs;

#include "ShapeInferenceOpInterfaces.cpp.inc"

namespace {

struct ShapeInferencePass : public mlir::PassWrapper<ShapeInferencePass, OperationPass<obs::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

    void runOnOperation() override {
        auto f = getOperation();

        llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist ;

        f.walk([&](mlir::Operation *op) {
            if (returnsDynamicShape(op)) {
                opWorklist.insert(op);
            }
        });

        while (!opWorklist.empty()) {
            auto nextop = llvm::find_if(opWorklist, allOperandsInferred);

            if (nextop == opWorklist.end()) {
                break;
            }

            Operation *op = *nextop;
            opWorklist.erase(op);

            LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
            if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
                shapeOp.inferShapes();
            } else {
                op -> emitError("unable to infer shape of operation without shape "
                      "inference interface");
                return signalPassFailure();
            }
        }

        if ( !opWorklist.empty()) {
            f->emitError("Shape inference failed, ")
               << opWorklist.size() << " operations couldn't be inferred\n";
        }
    }

    static bool returnsDynamicShape(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type resultType) {
            return !llvm::isa<RankedTensorType>(resultType);
        });
    }

    static bool allOperandsInferred(Operation *op) {
        return llvm::any_of(op->getOperandTypes(), [](Type operandType) {
            return llvm::isa<RankedTensorType>(operandType);
        });
    }

};

} //namespace

std::unique_ptr<mlir::Pass> mlir::obs::createShapeInferencePass() {
    return std::make_unique<ShapeInferencePass>();
}







