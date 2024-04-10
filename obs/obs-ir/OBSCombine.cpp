
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "Dialect.h"
using namespace mlir;
using namespace obs;

namespace {
#include "OBSCombine.inc"
} //namespace

struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
    SimplifyRedundantTranspose(mlir::MLIRContext *context) : 
        mlir::OpRewritePattern<TransposeOp>(context, /* benefit = */ 1) {}

    mlir::LogicalResult matchAndRewrite(TransposeOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::Value transposeInput = op.getOperand();
        TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

        if (!transposeInputOp) {
            return mlir::failure();
        }

        rewriter.replaceOp(op, {transposeInputOp.getOperand()});
        return success();
    }

};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context ) {
    results.add<SimplifyRedundantTranspose>(context);
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern, 
                FoldConstantReshapeOptPattern>(context);
}

