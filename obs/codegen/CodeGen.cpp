
#include "MLIRGen.h"

#include "AST.h"
#include "Dialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <clang/AST/Decl.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>
#include <cstdint>
#include <cstdlib>
#include <functional>

#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include <iostream>
#include <mlir/IR/Types.h>
#include <numeric>
#include <optional>
#include <sys/_types/_int32_t.h>
#include <vector>

using llvm::ArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;

#include "CodeGen.h"

using namespace clang;

namespace mlir {
namespace obs {

void MLIRGenImpl::dump() { MLIRModule->dump(); }

bool MLIRGenImpl::VisitFunctionDecl(clang::FunctionDecl *funcDecl) {
  auto location = loc(funcDecl->getBeginLoc());
  // Firstly, construct parameters.
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (auto *param : funcDecl->parameters()) {
    if (param->getType().getAsString() == "int") {
      argTypes.push_back(MLIRBuilder.getI32Type());
    } else {
      mlir::emitError(location) << "Parameter type (" +
                                       param->getType().getAsString() +
                                       ") are not supported at the moment.";
      exit(1);
    }
  }

  llvm::SmallVector<mlir::Type, 4> returnType;
  if (funcDecl->getReturnType().getAsString() == "int") {
    returnType.push_back(MLIRBuilder.getI32Type());
  }

  auto funcType = MLIRBuilder.getFunctionType(argTypes, returnType);

  MLIRBuilder.setInsertionPointToEnd(MLIRModule.getBody());

  auto funcOp = MLIRBuilder.create<mlir::obs::FuncOp>(
      location, funcDecl->getName(), funcType);

  MLIRBuilder.setInsertionPointToStart(&funcOp.front());

  return true;
}

bool MLIRGenImpl::VisitCompoundStmt(clang::CompoundStmt *compoundStmt) {
  return true;
}

bool MLIRGenImpl::VisitBinaryOperator(clang::BinaryOperator *binaryOperator) {
  switch (binaryOperator->getOpcode()) {
  case clang::BO_Assign: {
    VContext.setReading();
    TraverseStmt(binaryOperator->getRHS());
    VContext.resetReading();
    VContext.setWriting();
    TraverseStmt(binaryOperator->getLHS());
    VContext.resetWriting();
    break;
  }
  default:
    break;
  }
  return true;
}

bool MLIRGenImpl::VisitDeclRefExpr(clang::DeclRefExpr *declRef) {
  auto location = loc(declRef->getLocation());
  for (auto *Expr = VisitedExprs.begin(); Expr != VisitedExprs.end(); Expr++) {
    if (*Expr == declRef) { // Already being visited.
      // VisitedExprs.erase(Expr);
      return true;
    }
  }
  VisitedExprs.push_back(declRef);
  if (VContext.isWriting()) {
    Value value = SymbolTable.lookup(declRef->getDecl()->getName());
    MLIRBuilder.create<WriteOp>(location, value);
    return true;
  }
  if (VContext.isReading()) {
    Value value = SymbolTable.lookup(declRef->getDecl()->getName());
    MLIRBuilder.create<ReadOp>(location, value);
    return true;
  }
  return true;
}

bool MLIRGenImpl::VisitDeclStmt(clang::DeclStmt *stmt) { return true; }

bool MLIRGenImpl::VisitVarDecl(clang::VarDecl *varDecl) {
  auto location = loc(varDecl->getBeginLoc());
  Expr *init = varDecl->getInit();
  if (init && init->getType().getAsString() == "int") {
    VContext.setReading();
    TraverseStmt(init);
    VContext.resetReading();
    StringRef typeName = "INT";
    auto ownType =
        mlir::obs::OwnType::get(MLIRModule.getContext(), typeName, {1});

    std::vector<int32_t> data;
    data.push_back(1);
    auto value = MLIRBuilder.create<OwnOp>(location, ownType, typeName,
                                           llvm::ArrayRef<int32_t>(data));
    SymbolTable.insert(varDecl->getName(), value);
  }

  return true;
}

mlir::ModuleOp MLIRGenImpl::mlirGen(clang::TranslationUnitDecl &decl) {
  ScopedHashTableScope<StringRef, mlir::Value> varScope(SymbolTable);
  MLIRModule = mlir::ModuleOp::create(MLIRBuilder.getUnknownLoc());
  TraverseDecl(&decl);

  if (failed(mlir::verify(MLIRModule))) {
    MLIRModule->emitError("OBS module verification error");
    return nullptr;
  }
  return MLIRModule;
}

bool MLIRGenImpl::VisitIfStmt(clang::IfStmt *IfStmt) {
  auto location = loc(IfStmt->getBeginLoc());
  auto branch = MLIRBuilder.create<NonDeterBranch>(location);

  auto InsertionPoint = MLIRBuilder.saveInsertionPoint();
  auto *block = MLIRBuilder.createBlock(&branch.getOptions());

  VContext.setReading();
  TraverseStmt(IfStmt->getCond());
  VContext.resetReading();

  auto *blockThen = MLIRBuilder.createBlock(&branch.getOptions());

  TraverseStmt(IfStmt->getThen());

  auto *blockElse = MLIRBuilder.createBlock(&branch.getOptions());
  TraverseStmt(IfStmt->getElse());

  SmallVector<Block *, 2> SuccBlocks;
  SuccBlocks.push_back(blockThen);
  SuccBlocks.push_back(blockElse);

  MLIRBuilder.setInsertionPoint(block, block->end());
  MLIRBuilder.create<SuccOp>(location, llvm::ArrayRef<Block *>(SuccBlocks));

  auto *exitBlock = MLIRBuilder.createBlock(&branch.getOptions());
  MLIRBuilder.create<SkipOp>(location);
  MLIRBuilder.setInsertionPoint(blockThen, blockThen->end());
  MLIRBuilder.create<SuccOp>(location, exitBlock);
  MLIRBuilder.setInsertionPoint(blockElse, blockElse->end());
  MLIRBuilder.create<SuccOp>(location, exitBlock);
  MLIRBuilder.getBlock();
  MLIRBuilder.restoreInsertionPoint(InsertionPoint);

  return true;
}

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          clang::ASTContext &clang_context) {
  return MLIRGenImpl(context, clang_context)
      .mlirGen(*clang_context.getTranslationUnitDecl());
}

} // namespace obs
} // namespace mlir
