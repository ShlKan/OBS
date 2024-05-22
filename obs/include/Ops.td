
#ifndef OBS_OPS
#define OBS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "ShapeInferenceInterface.td"

def OBS_Dialect: Dialect {
    //The namespace of the Dialect.
    let name = "obs";

    let summary = "A high-level dialect for analyzing and optimzing the Language";

    let description = [{
        The Toy language is a tensor-based language that allows you to define functions, perform some math computation, and
        print results. This dialect provides a reprentation of the language that is amenable to analysis and optimization.
    }];

    let useDefaultTypePrinterParser = 1;

    let cppNamespace = "::mlir::obs";
}

class OBS_Op<string mnemonic, list<Trait> traits = []> : Op<OBS_Dialect, mnemonic, traits>;

def OBS_StructType : 
    DialectType<OBS_Dialect, CPred<"::llvm::isa<StructType>($_self)">, "OBS struct type">;

def OBS_Type : AnyTypeOf<[F64Tensor, OBS_StructType]>;

def ConstantOp : OBS_Op<"constant", [Pure]> {

    let summary = "constant operation" ;

    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached to the operation as an attribute.
    }];


    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "DenseElementsAttr":$value), 
        [{build($_builder, $_state, value.getType(), value);}] >,
        OpBuilder<(ins "double":$value)>
    ];

}

def AddOp : OBS_Op<"add", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two tensors.
        The shape of the tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}  

def FuncOp : OBS_Op<"func", 
        [FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "user defined function operation";
    let description = [{
        The "obs.func" operation represents a user defined function. These are callable SSA-region operations
        that contain obs computations.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                                CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
                    >];

    let extraClassDeclaration = [{
        ArrayRef<Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        ArrayRef<Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        Region *getCallableRegion() {
            return &getBody();
        }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def StructAccessOp : OBS_Op<"struct_access", [pure]> {
    let summary = "struct access";
    let description = [{
        Access the Nth element of a value returning a struct type.
    }];

    let arguments = (ins OBS_StructType:$input, I64Attr:$index);
    let results = (outs OBS_Type:$output);

    let assemblyFormat = [{
        $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
        OpBuilder<(ins "Value":$input, "size_t":$index)>
    ];

    let hasVerifier = 1;

    let hasFolder = 1;

}

def StructConstantOp : OBS_Op<"struct_constant", [ConstantLike, Pure]> {
    let summary = "struct constant";
    let description = [{
        Constant operation turns a literal struct value into an SSA value. The data
        is attached to the operation as an attribute. The struct constant is encoded
         as an array of other constant values. For example:

        ```mlir
            %0 = toy.struct_constant [
            dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
            ] : !toy.struct<tensor<*xf64>>
        ```
    }];

    let arguments = (ins ArraryAttr:$value);
    let results = (outs OBS_StructType:$output);

    let assemblyFormat = "$value attr-dict `:` type($output)";

    let hasVerifier = 1;
    let hasFolder = 1;
}

def GenericCallOp: OBS_Op<"generic_call", 
                          [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "generic call operation";
    let description = [{
        Generic calls represent calls to a user defined function that needs to be specialized for the shape
        of its arguments.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<OBS_Type>:$inputs);

    let results = (outs OBS_Type);

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    let builders = [
        OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
    ];
}

def MulOp : OBS_Op<"mul", [Pure, 
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise multiplication operation";

    let description = [{
        The "mul" operation performs element-wise multiplication between two
        tensors. The shapes of the tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def PrintOp: OBS_Op<"print"> {
    let summary = "print operation";
    let description = [{
        The print builtin operation prints a given input sensor, and produces no results.
    }];
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReshapeOp: OBS_Op<"reshape", [Pure]> {
    let summary = "tensor reshape operation";
    let description = [{
        Reshape operation is transforming its input sensor into a new tensor with the same number of
        elements but different shapes.
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs StaticShapeTensorOf<[F64]>);

    let hasCanonicalizer = 1;

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];
}

def TransposeOp: OBS_Op<"transpose", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "transpose operation";
    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

def ReturnOp : OBS_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
    let summary = "return operation";
    let description = [{
        The "return" operation represents a return operation within a function;
    }];

    let arguments = (ins Variadic<OBS_Type>:$input);
    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
 
    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];

    let extraClassDeclaration = [{
        bool hasOperand() {
            return getNumOperands() != 0;
        }
    }];

    let hasVerifier = 1;
}


def CastOp : OBS_Op <"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    Pure,
    SameOperandsAndResultShape
]> {
    let summary = "shape cast operation";

    let description = [{
        The "cast" operation converts a tensor from one type to an equivalent type
        without changing any data elements.
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor:$output);

    let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}


#endif //OBS_OPS
