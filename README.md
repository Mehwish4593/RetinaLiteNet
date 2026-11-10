# RetinaLiteNet

A lightweight multitask network for retinal feature segmentation. A single
encoder feeds a transformer block at the bottleneck, and a decoder with skip
connections and CBAM attention produces two maps at once: blood vessels and the
optic disc. 66,194 parameters in the published configuration, 0.25 MB of
weights and 2.46 GFLOPs.

> M. Mehmood, M. Alsharari, S. Iqbal, I. Spence, M. Fahim, *RetinaLiteNet: A
> Lightweight Transformer based CNN for Retinal Feature Segmentation*,
> IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops
> (CVPRW), 2024, pp. 2454-2463.

## Contents

    retinalitenet_drive.py          training run on DRIVE, and the model definition
    retinalitenet_iostar.py         training run on IOSTAR
    MTLTransfuseep300bs16head4.h5   DRIVE weights
    Model_weights_IOSTAR.h5         IOSTAR weights
    FLOPs vs inference.ipynb        parameter, FLOP and timing measurements

## Read this before loading the weights

The two released checkpoints were trained with different attention widths.
`num_heads` is 4 in both, but `key_dim` is not, and the wrong value fails with a
shape mismatch on `multi_head_attention/query/kernel`.

| checkpoint | `key_dim` | parameters |
|---|---|---|
| `MTLTransfuseep300bs16head4.h5` (DRIVE) | **16** | 57,810 |
| `Model_weights_IOSTAR.h5` (IOSTAR)      | **32** | 66,194 |

Both scripts hardcode `key_dim=32`, at line 262 of `retinalitenet_drive.py` and
line 248 of `retinalitenet_iostar.py`. That matches the IOSTAR checkpoint. To
load the DRIVE checkpoint, change that line to `key_dim=16` first.

The DRIVE checkpoint is therefore the `head 4, dk 16` configuration from the
paper's Table 3 ablation, not the `head 4, dk 32` configuration the headline
numbers come from. Its 57,810 parameters match the 0.057M that Table 4 lists
against `4 / 16`. The weights for the headline DRIVE configuration, which
`retinalitenet_drive.py` would save as `MTLTransfuseep300bs16head4_32.h5`, are
not in this repository. The IOSTAR checkpoint is the full `dk 32`
configuration.

## Using the model

The builder is `create_transfuse_net(input_shape)`, with `cbam_block`,
`channel_attention` and `spatial_attention` just above it. Both scripts are
exported notebooks that run a full training job from top to bottom, so copy
those four functions into a module of your own rather than importing a script.

    model = create_transfuse_net((512, 512, 3))
    model.load_weights("Model_weights_IOSTAR.h5")
    bv, od = model.predict(x)

`final_output1` is the blood-vessel map and `final_output2` the optic disc,
both sigmoid activated at the input resolution.

Preprocessing follows `get_data()` in the scripts: `skimage.transform.resize`
to 512x512 with `preserve_range`, then divide by 255. skimage anti-aliases when
downsampling and PIL's `BILINEAR` does not, so the two are not interchangeable
on DRIVE's 584x565 inputs.

The weights are Keras 2 `.h5` files, so build the model under `tf_keras`
(legacy Keras) rather than Keras 3.

## Results

DRIVE, the 20 official test images with the first observer's vessel
annotations. Per-image threshold maximising F1 against that image's own mask,
then `bwareaopen(out, 50)`, measured over the full frame with no field-of-view
mask.

| task | Dice | Jaccard | Sen. | Spe. |
|---|---|---|---|---|
| Blood vessels, 8-way TTA | **79.53** | 66.05 | 78.05 | 98.26 |
| Blood vessels, single pass | 78.87 | 65.14 | 76.12 | 98.39 |
| Optic disc, 8-way TTA | **93.79** | 88.44 | 95.58 | 99.82 |
| Optic disc, single pass | 92.81 | 86.86 | 93.00 | 99.84 |

For reference, Table 3 of the paper reports 79.3 Dice and 66.4 Jaccard for
vessels and 89.1 and 81.9 for the optic disc at `head 4, dk 16`, and 80.6 /
67.5 and 93.3 / 88.0 for the headline `head 4, dk 32`. The vessel figures above
reproduce the `dk 16` row, which is the configuration these weights hold. The
optic disc figures come out above both.

One caveat on quoting these Dice figures. The threshold is chosen per image
against that image's own ground truth. That is the convention in much of the
retinal segmentation literature and is what the table follows, but it is an
upper bound rather than a deployment figure, since a running system has no mask
to tune against.

## Architecture

**Encoder.** Three stages, each a 3x3 convolution with ReLU, then 2x2
max-pooling and batch normalisation. Widths 8, 16 and 32, so a 512x512 input
reaches the bottleneck at 64x64x32.

**Transformer block.** The bottleneck is flattened to a sequence of 4096 tokens
of 32 channels, passed through 4-head self-attention, averaged over the
sequence and tiled back across the 64x64 grid. No positional encoding; Table 5
of the paper shows it makes little difference here.

**CBAM.** Channel attention (a shared two-layer MLP over the global average and
max pools, summed, sigmoid) followed by spatial attention (a 7x7 convolution
over the channel-wise mean and max, sigmoid). Applied to the tiled transformer
output and to each of the first two decoder stages.

**Decoder.** Three stride-2 transposed convolutions. The first two concatenate
a CBAM-attended feature map with the matching encoder stage and follow with a
3x3 convolution. Two 1x1 sigmoid heads give the vessel and optic disc maps.

## Training

The scripts are the original runs: 300 epochs, Adam at its default learning
rate, and Dice plus IoU loss on each head. `get_data()` expects a folder
holding `images/`, `GT_BV/` and `GT_OD/`, and both datasets were augmented to
400 training images.

Two things to know if you retrain. The 80/20 split is taken from the augmented
pool, so augmented copies of the same fundus image land on both sides and the
validation score measures fit to transformations of the training images rather
than generalisation; judge a checkpoint on the held-out test set instead. And
the IoU term in the loss thresholds its input at 0.5, which carries no
gradient, so it moves the reported loss and the choice of best checkpoint but
not the weights.

## Citation

    @inproceedings{mehmood2024retinalitenet,
      title     = {RetinaLiteNet: A Lightweight Transformer based CNN for
                   Retinal Feature Segmentation},
      author    = {Mehmood, Mehwish and Alsharari, Majed and Iqbal, Shahzaib and
                   Spence, Ivor and Fahim, Muhammad},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
                   and Pattern Recognition (CVPR) Workshops},
      pages     = {2454--2463},
      year      = {2024}
    }

## Licence

MIT, see LICENSE.
