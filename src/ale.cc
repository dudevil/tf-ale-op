#include <ale_interface.hpp>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/util/guarded_philox_random.h"

using namespace tensorflow;

REGISTER_OP("Ale")
    .Attr("rom_file: string")
    .Attr("frameskip_min: int = 2")
    .Attr("frameskip_max: int = 5")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Input("action: int32")
    .Output("reward: float")
    .Output("done: bool")
    .Output("screen: uint8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      // c->set_output(2, c->MakeShape(ale.getScreen().height(), ale.getScreen().width(), 3));
      return Status::OK();
    });


class AleOp : public OpKernel {
 public:
  explicit AleOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("rom_file", &rom_file_));
    auto full_rom_path_ = ROM_PATH + rom_file_;
    OP_REQUIRES_OK(context,
		   Env::Default()->FileExists(full_rom_path_));

    ale_.loadROM(full_rom_path_);
    OP_REQUIRES_OK(context,
                   context->GetAttr("frameskip_min", &frameskip_min_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("frameskip_max", &frameskip_max_));
    OP_REQUIRES(context, frameskip_min_ > 0,
                errors::InvalidArgument("frameskip_min must be > 0"));
    OP_REQUIRES(context, frameskip_max_ >= frameskip_min_,
                errors::InvalidArgument("frameskip_max must be >= frameskip_min"));

    auto legalActionsVec = ale_.getLegalActionSet();
    std::copy(legalActionsVec.begin(), legalActionsVec.end(), std::inserter(legalActions_, legalActions_.end()));
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input_tensor.shape()),
                errors::InvalidArgument("Ale expects scalar action."));
    auto input = input_tensor.scalar<int32>();
    Action action = (Action) input(0);
    OP_REQUIRES(context, legalActions_.find(action) != legalActions_.end(),
                errors::InvalidArgument("Action is out of legal actions range."));

    auto w = ale_.getScreen().width();
    auto h = ale_.getScreen().height();
    std::vector<unsigned char> screen_buff;
    screen_buff.reserve(w * h * 3);

    auto local_gen = generator_.ReserveSamples32(1);
    random::SimplePhilox random(&local_gen);
    auto to_repeat = frameskip_min_ + random.Uniform(frameskip_max_ - frameskip_min_);

    float r = 0.0;
    for(;to_repeat > 0; --to_repeat){
      r += ale_.act(action);
    }

    auto done = ale_.game_over();
    ale_.getScreenRGB(screen_buff);

    if(done) ale_.reset_game();
    
    Tensor* reward_tensor = NULL;
    Tensor* done_tensor = NULL;
    Tensor* screen_tensor = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(),
                                                     &reward_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape(),
						     &done_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, {h, w, 3},
						     &screen_tensor));

    auto output_r = reward_tensor->scalar<float>();
    auto output_d = done_tensor->scalar<float>();
    auto output_s = screen_tensor->flat<unsigned char>();
    
    output_r(0) = r;
    output_d(0) = done;
    std::copy_n(screen_buff.begin(), screen_buff.size(),
		output_s.data()); // get rid of copy?
  }

private:
  ALEInterface ale_;
  std::set<Action> legalActions_;
  std::string rom_file_;
  int frameskip_min_;
  int frameskip_max_;
  GuardedPhiloxRandom generator_;
};

REGISTER_KERNEL_BUILDER(Name("Ale").Device(DEVICE_CPU), AleOp);
