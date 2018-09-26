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
