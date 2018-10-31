# Note

## 01

batch-size: 12
duration: 8
lr: 1e-4 with cosine restarts and exponential decay

## 03

steps: 127k
lr: 1e-4 from 0-63.5k, decay to 0 from 63.5-127k

## 04

steps: 44k
lr: 1e-4 from 0-22k, decay to 0 from 22-44k

## 05

steps: 32k
added activation before patch critic
domain label (shape): (N, 1) => (N,)

## 06

steps: 128k

## 07

steps: 32k
(unchanged) discriminator: InstanceNorm => None

## 08

steps: 16k
generator: InstanceNorm, adv loss
discriminator: InstanceNorm, adv loss
- kernel1: [1, 8] => [1, 4]
- patch critic: [1, 1] => [1, 3]

## 09

steps: 16k
generator: InstanceNorm, adv loss
discriminator: InstanceNorm, adv loss, cls loss

## 10

steps: 16k
generator: InstanceNorm, adv loss, cls loss
discriminator: InstanceNorm, adv loss, cls loss

## 11

steps: 16k
generator: InstanceNorm
discriminator: InstanceNorm

## 12

steps: 16k
generator: InstanceNorm
discriminator: NoNorm

## 13

steps: 16k
lr: 1e-4 in 0-4k, decay to 1e-5 in 4-12k, 1e-5 in 12-16k

## 14

steps: 32k
lr: 1e-4 in 0-8k, decay to 1e-5 in 8-24k, 1e-5 in 24-32k

## 15

steps: 16k
Discriminator - InBlock - stride: [1, 2] => [1, 1]

## 16

steps: 16k
(unchanged)
Discriminator - EBlock_1 - stride: [1, 2] => [1, 1]

## 17

steps: 64k
Discriminator - DenseNet

## 18

steps: 64k
lr: 5e-4 in 0-16k, decay to 5e-5 in 16-48k, 5e-5 in 48-64k
Discriminator - SpeakerRecognition(153) with original network

## 19

steps: 16k
lr: 5e-4 in 0-8k, decay to 0 in 8-16k
Discriminator - SpeakerRecognition(153) with original network

## 20

steps: 16k
lr: 5e-4 in 0-8k, decay to 0 in 8-16k
Discriminator - SpeakerRecognition(152) with DenseNet network
batch size: 12 => 6

## 21

steps: 64k
lr: 2e-4 in 0-32k, decay to 0 in 32-64k
Discriminator - SpeakerRecognition(153) with original network

## 22

steps: 64k
lr: 1e-4 in 0-32k, decay to 0 in 32-64k
Discriminator - SpeakerRecognition(153) with original network

## 23

steps: 32k
lr: 5e-4 in 0-8k, decay to 0 in 8-32k
Discriminator - SpeakerRecognition(153) with original network

## 24

steps: 32k
lr: 5e-4 in 0-8k, decay to 0 in 8-32k
Discriminator - VGG-like network

## 25

steps: 32k
lr: 5e-4 in 0-8k, decay to 0 in 8-32k
Discriminator - SpeakerRecognition(155) with VGG-like network

## 26

steps: 128k
lr: 5e-4 in 0-32k, decay to 0 in 32-128k
Discriminator - SpeakerRecognition(153) with original network

## 27

steps: 32k
lr(g): 5e-4 in 0-8k, decay to 0 in 8-32k
lr(d): 2e-4 in 0-8k, decay to 0 in 8-32k

## 28

steps: 96k
lr: 5e-4 in 0-24k, decay to 0 in 24-96k

## 29

steps: 32k
lr(g): 2e-4 in 0-8k, decay to 0 in 8-32k
lr(d): 5e-4 in 0-8k, decay to 0 in 8-32k

## 30

steps: 96k
lr: 1e-4

## 31

steps: 96k
lr: 1e-4
Nadam - beta1: 0.9 => 0.5

## 32

steps: 96k
lr: 1e-4
Adam: beta1=0.5

## 33

steps: 96k
lr: 2e-4
Adam: beta1=0.5

## 34

steps: 96k
lr: 2e-4
rec loss: MS-SSIM
MS-SSIM weights: 0.1, 0.15, 0.2, 0.25, 0.3

## 35

steps: 96k
lr: 1e-4 in 0-24k, decay to 0 in 24-96k
fixed negative values in MS-SSIM
MS-SSIM weights: default

## 36

rec loss: 1.0*L1 + MS-SSIM

## 37

patch-based classification loss

## 38

rec loss: MS-SSIM
fixed issues that MS-SSIM is not in loss collection

## 39

sample-based classification loss
