set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )
  
include(CMakeDependentOption)
include(GNUInstallDirs)

set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --src-root
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --includedir
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})


include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)

# Make sure that our source directory is on the current cmake module path so
# that we can include cmake files from this directory.
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

include(AddOBS)

include_directories(include/)

set(LLVM_TARGET_DEFINITIONS obs-ir/OBSCombine.td)
mlir_tablegen(OBSCombine.inc -gen-rewriters)
add_public_tablegen_target(OBSCombine)

add_obs_executable(obstest 
    obs-ir/obs.cpp 
    obs-ir/Dialect.cpp
    obs-ir/MLIRGen.cpp
    obs-ir/OBSCombine.cpp
    obs-ir/ShapeInferencePass.cpp
    obs-ir/LowerToAffineLoops.cpp
    obs-ir/LowerToLLVM.cpp
    parser/AST.cpp
  DEPENDS
    OBSOpGen
    OBSCombine
    ShapeInferenceInterfaceIncGen
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(obstest
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs} 
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransforms
)

