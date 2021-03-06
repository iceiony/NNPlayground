library(ggplot2)
library(plyr)
library(dplyr)

mean_errors <- function(x){
    res <-  ldply(x, function(x){
                    data.frame(mean = mean(x), sd = sd(x))
            })
    res$.id = seq(nrow(res))
    res$se = res$sd/sqrt(nrow(res))
    res
}

dynet   <- read.csv('./dynet/cpu.csv')   %>% mean_errors
#pyTorch <- read.csv('./pytorch/cpu.csv') %>% mean_errors
#chainer <- read.csv('./chainer/cpu.csv') %>% mean_errors
dynet_gpu   <- read.csv('./dynet/gpu.csv')   %>% mean_errors
#pyTorch_gpu <- read.csv('./pytorch/gpu.csv') %>% mean_errors
#chainer_gpu <- read.csv('./chainer/gpu.csv') %>% mean_errors

dynet$type   = 'dynet'
#pyTorch$type = 'pyTorch'
#chainer$type = 'chainer' 
dynet_gpu$type   = 'dynet-gpu'
#pyTorch_gpu$type = 'pyTorch-gpu'
#chainer_gpu$type = 'chainer-gpu'

#dat <- rbind(dynet, pyTorch, chainer)
#dat <- rbind(dynet_gpu, pyTorch_gpu, chainer_gpu)
dat <- rbind(dynet, dynet_gpu)
#dat <- rbind(pyTorch, pyTorch_gpu)
#dat <- rbind(chainer, chainer_gpu)

# Make the plot
plt <- ggplot(data = dat, aes(x = .id, y = mean, ymin = mean - se, ymax = mean + se, fill = type, linetype = type)) + 
        geom_line() + 
        geom_ribbon(alpha=0.5) +
        ylab('Learning error') +
        xlab('itteration')
print(plt)
ggsave('cpu.png')
