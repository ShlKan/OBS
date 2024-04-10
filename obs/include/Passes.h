#ifndef OBS_PASSES_H
#define OBS_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace obs {
    std::unique_ptr<Pass> createShapeInferencePass();
} //namespace obs

} //namespace mlir

#endif //OBS_PASSES_H