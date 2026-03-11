#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/DTensorState.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_functional_assert_async.h>
#include <ATen/ops/add.h>
#include <ATen/ops/all.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/one_hot_native.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

Tensor one_hot(const Tensor &self, int64_t num_classes) {
    TORCH_CHECK(self.dtype() == kLong, "one_hot is only applicable to index tensor of type LongTensor.");

    // using meta bit test to catch Fake Tensor as well until __torch_function__
    if (self.key_set().has_all(DispatchKeySet(BackendComponent::MetaBit)) ||
            self.key_set().has_all(DispatchKeySet(DispatchKey::Python))) {
        // functional version that torch.compiles better and works with dynamic shapes
        if (num_classes == -1) {
          num_classes = self.max().item().toLong() + 1;
        }
        // Validate index bounds using _functional_assert_async so the
        // assertions survive functionalization and appear in the compiled graph.
        // _assert_async (non-functional, void-returning) gets dropped during
        // CIA decomposition tracing because proxy tracing doesn't capture
        // void-returning ops.
        auto dep = at::zeros({}, self.options().dtype(kLong));
        dep = at::_functional_assert_async(at::all(at::ge(self, 0)),
            "one_hot: Class values must be non-negative.", dep);
        dep = at::_functional_assert_async(at::all(at::lt(self, num_classes)),
            "one_hot: Class values must be smaller than num_classes.", dep);
        // Use scatter which gets bounds checking through indirect indexing
        // (check_bounds), supported on all backends including MPS.
        auto shape = self.sym_sizes().vec();
        shape.emplace_back(num_classes);
        at::Tensor ret = at::zeros_symint(shape, self.options());
        ret.scatter_(-1, self.unsqueeze(-1), 1);
        // dep is always zero; adding it creates a data dependency on the
        // assertions, preventing dead code elimination in the compiled graph.
        return at::add(ret, dep);
    }

    auto shape = self.sym_sizes().vec();

    // empty tensor could be converted to one hot representation,
    // but shape inference is not possible.
    if (self.sym_numel() == 0) {
        if (num_classes <= 0) {
            TORCH_CHECK(false, "Can not infer total number of classes from empty tensor.");
        } else {
            shape.emplace_back(num_classes);
            return at::empty_symint(shape, self.options());
        }
    }

    // non-empty tensor
    if (self.device().type() != at::kCUDA && self.device().type() != at::kMPS &&
        self.device().type() != at::kPrivateUse1 && self.device().type() != at::kXLA) {
      // for cuda, rely on device assert thrown by scatter
      TORCH_CHECK(self.min().item().toLong() >= 0, "Class values must be non-negative.");
    }
    if (num_classes == -1) {
        num_classes = self.max().item().toLong() + 1;
    } else {
        if (self.device().type() != at::kCUDA && self.device().type() != at::kMPS &&
            self.device().type() != at::kPrivateUse1 && self.device().type() != at::kXLA) {
          // rely on device asserts from scatter to avoid sync here
          TORCH_CHECK(num_classes > self.max().item().toLong(), "Class values must be smaller than num_classes.");
        } else {
            //for cuda, assert that num_classes is at least 1
            TORCH_CHECK(num_classes >= 1, "num_classes should be positive");
        }
    }

    shape.emplace_back(num_classes);
    Tensor ret = at::zeros_symint(shape, self.options());
    ret.scatter_(-1, self.unsqueeze(-1), 1);
    return ret;
}

} // namespace at::native
