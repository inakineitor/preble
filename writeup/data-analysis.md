# Data Analysis

## Meanings

`X`: The fastest naive method.

## Show that we made a scheduler that is faster than Preble

### Make a simulator that supports a perfect oracle

1. [ ] AAAAAAAA

### Make a scheduler that takes in the length data to make predictions

1. [ ] AAAAAAAA

## Show that Preble does not scale as desired (under many settings)

1. [x] Show that `X` (or whatever other method is better) outperforms other naive methods for low and high RPS. We then only compare against this method to make the simulations faster to run. PLOT 1
1. [ ] Show that `Preble` does not run better than `X` for high RPS under a diverse configurations:
   1. [ ] Multiple different numbers of included workloads
   1. [x] Multiple different numbers of GPUs. PLOT 2
   1. [x] Multiple different request length distributions. PLOT 3

## Explain the reason for why Preble does not scale

1. [ ] Analyze overhead
1. [ ] (potentially) Run the simulator without overhead and show that Preble beats the naive ones. This can be done by either:
   1. [ ] Subtracting the overhead from the latency while plotting (easier)
   1. [ ] Zero out the overhead time within the simulator (harder)

## Give a reason for why the proposed new solution is more scalable to higher RPS than the current one

1. [ ] (optional) Do some analysis that defends our point?
