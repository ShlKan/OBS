
#ifndef _OBS_CODEGEN_H
#define _OBS_CODEGEN_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/SourceLocation.h>
#include <memory>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "Dialect.h"
#include <mlir-c/IR.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

namespace mlir {
namespace obs {

struct VisitContext {
private:
  bool IsReading;
  bool IsWriting;

public:
  void setReading() { IsReading = true; }
  void resetReading() { IsReading = false; }
  bool isReading() { return IsReading; }
  void setWriting() { IsWriting = true; }
  void resetWriting() { IsWriting = false; }
  bool isWriting() { return IsWriting; }
  void initContext() {
    IsWriting = false;
    IsReading = false;
  }
};

class MLIRGenImpl : public clang::RecursiveASTVisitor<MLIRGenImpl> {
public:
  MLIRGenImpl(mlir::MLIRContext &mlir_context, clang::ASTContext &clang_context)
      : MLIRBuilder(&mlir_context), ClangASTContext(clang_context) {
    VContext.initContext();
  }

  bool VisitFunctionDecl(clang::FunctionDecl *funcDecl);
  bool VisitCompoundStmt(clang::CompoundStmt *compoundStmt);
  bool VisitBinaryOperator(clang::BinaryOperator *binaryOperator);
  bool VisitDeclStmt(clang::DeclStmt *stmt);
  bool VisitVarDecl(clang::VarDecl *varDecl);
  bool VisitDeclRefExpr(clang::DeclRefExpr *declRef);
  bool VisitIfStmt(clang::IfStmt *IfStmt);

  void dump();

  mlir::ModuleOp mlirGen(clang::TranslationUnitDecl &decl);

private:
  mlir::ModuleOp MLIRModule;
  mlir::OpBuilder MLIRBuilder;
  clang::ASTContext &ClangASTContext;
  struct VisitContext VContext;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> SymbolTable;
  llvm::SmallVector<clang::Expr *, 4> VisitedExprs;

  mlir::Location loc(const clang::SourceLocation &loc) {
    // FIXME: use clang locatoin.
    clang::SourceManager &SM = ClangASTContext.getSourceManager();
    clang::PresumedLoc PLoc = SM.getPresumedLoc(loc);
    return mlir::FileLineColLoc::get(
        MLIRBuilder.getStringAttr(PLoc.getFilename()), PLoc.getLine(),
        PLoc.getColumn());
  }
};

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          clang::ASTContext &clang_context);

} // namespace obs
} // namespace mlir

#endif