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

