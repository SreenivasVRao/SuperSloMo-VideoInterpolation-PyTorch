#!/usr/bin/env python3


def validate_forward_pass_inputs(forward_pass_func):
    def func_wrapper(self, input_images, target_images, split, iteration, t_interp):
        assert input_images.shape[1] == self.cfg.getint("TRAIN", "N_FRAMES")
        assert target_images.shape[1] == t_interp.shape[1] == self.cfg.getint("TRAIN", "N_FRAMES") -1

        assert bool(
            (t_interp > 0).all() and (t_interp < 1).all()
        ), "Interpolation values out of bounds."

        return forward_pass_func(self, input_images, target_images, split, iteration, t_interp)

    return func_wrapper


def validate_sampling(get_dataset_func):
    def func_wrapper(config, split):
        if config.getboolean("EVAL", "EVAL_MODE"):
            assert config.get("DATALOADER", "T_SAMPLE") == "NIL"
        else:
            assert config.get("DATALOADER", "T_SAMPLE") != "NIL"

        return get_dataset_func(config, split)

    return func_wrapper


def validate_image_paths_length(func):
    def func_wrapper(self, img_paths):
        assert len(img_paths) == self.WINDOW_LENGTH
        assert len(img_paths) >= self.reqd_images
        img_paths = func(self, img_paths)
        assert len(img_paths) == self.reqd_images, "Incorrect length of input sequence."
        return img_paths

    return func_wrapper


def validate_train_tensor_shapes(func):
    def func_wrapper(self, idx):
        input_tensor, target_tensor, t_interp = func(self, idx)
        assert (
            input_tensor.size()[0] == self.n_frames
        ), "%s frames doesn't match expected number" % (input_tensor.size()[0])

        assert (
            target_tensor.size()[0] == self.n_frames - 1
        ), "%s frames doesn't match expected number" % (target_tensor.size()[0])

        return input_tensor, target_tensor, t_interp

    return func_wrapper


def validate_inference_tensor_shapes(func):
    def func_wrapper(self, idx):
        input_tensor, target_tensor, n_targets = func(self, idx)
        assert (
            input_tensor.size()[0] == self.n_frames
        ), "%s frames doesn't match expected number" % (input_tensor.size()[0])

        if self.dataset_name == "VIMEO" and self.eval_mode:
            pass
        else:
            assert (
                target_tensor.size()[0] == self.interp_factor - 1
            ), "%s frames doesn't match expected number" % (target_tensor.size()[0])
        assert 0 < n_targets < self.interp_factor

        return input_tensor, target_tensor, n_targets

    return func_wrapper


def validate_inference_item_indexes(func):
    def func_wrapper(self):
        assert self.t_sample == "NIL"
        input_idx, groundtruth_idx = func(self)
        assert len(groundtruth_idx) == self.interp_factor - 1
        return input_idx, groundtruth_idx

    return func_wrapper


def validate_batch_crop_dimensions(crop_func):
    def func_wrapper(self, batch):
        assert batch.shape[1:] == (3, self.H_REF, self.W_REF), "Invalid shape"
        batch = crop_func(self, batch)
        assert batch.shape[1:4] == (self.H_IN, self.W_IN, 3,), "Dimensions are incorrect."

        return batch

    return func_wrapper


def validate_evaluation_interpolation_result(interpolation_func):
    def func_wrapper(self, current_batch):
        outputs = interpolation_func(self, current_batch)
        if not self.dataset == "VIMEO":
            assert len(outputs) == self.interp_factor - 1, "Wrong number of outputs."
        return outputs

    return func_wrapper


def validate_t_interp(t_interp_generator_func):
    def func_wrapper(self, mid_idx):
        t_interp = t_interp_generator_func(self, mid_idx)
        assert (t_interp > 0).all() and (t_interp < 1).all(), "Incorrect values."
        return t_interp

    return func_wrapper
