#ifndef MLIR_DIALECT_TRAITS_H
#define MLIR_DIALECT_TRAITS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ShapeInferenceInterface.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <mlir/IR/Types.h>

namespace mlir{
namespace obs {
namespace detail {
struct StructTypeStorage;
} //detail
} //obs
} //mlir

#include "Dialect.h.inc"

#define GET_OP_CLASSES
#include "Ops.h.inc"

namespace mlir {
namespace obs {

class StructType : public mlir::Type::TypeBase<StructType, mlir::Type, detail::StructTypeStorage> {

public:
    using Base::Base;

    static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

    llvm::ArrayRef<mlir::Type> getElementTypes();

    size_t getNumElementTypes() { return getElementTypes().size(); }

    static constexpr llvm::StringLiteral name = "obs.struct";
};
} //obs
} //mlir


#endif //MLIR_DIALECT_TRAITS_H
