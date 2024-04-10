#ifndef OBS_COMBINE
#define OBS_COMBINE

include "mlir/IR/PatternBase.td"
include "Ops.td"

def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<(ReshapeOp:$res $arg), (replaceWithValue $arg), 
                    [(TypesAreIdentical $res, $arg)]>;

def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;

def FoldConstantReshapeOptPattern : Pat<(ReshapeOp: $res (ConstantOp $arg)), 
                                            (ConstantOp (ReshapeConstant $arg, $res))>;

#endif //OBS_COMBINE