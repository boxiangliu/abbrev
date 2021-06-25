x = read.table("publication/rerank2.tsv", header=T)
x$features = factor(floor((x$pred - 1)/4))
library(cowplot)
library(ggplot2)
p = ggplot(x, aes(x=features, y=F)) + geom_boxplot() + theme_cowplot() + scale_y_continuous(name="F-score") + scale_x_discrete(name="Reranking Features", labels=c("rank\n", "rank +\ncharmatch", "rank +\ncharmatch +\nfreq")) + theme(axis.text.x = element_text(angle = 45, vjust=0.9, hjust=0.7), axis.title.x = element_text(margin = margin(t = -15, r = 0, b = 0, l = 0)))
save_plot("publication/fig_2.pdf", p, base_width=3, base_height=3.3)

x$lab = factor(x$lab, levels=c("BioADI", "SH", "Ab3P", "MEDSTRACT"))
p1 = ggplot(x, aes(x=lab, y=F, fill=as.factor(pred))) + geom_bar(stat="identity", position="dodge") + theme_cowplot() + scale_x_discrete(name=NULL, breaks=c("BioADI", "SH", "Ab3P", "MEDSTRACT")) + scale_y_continuous(name="F-score") + coord_cartesian(ylim=c(0.6,0.9)) + theme(axis.text.x = element_text(angle = 45, vjust=0.9, hjust=0.9), plot.margin=unit(c(5.5,5.5,5.5,5.5),"pt"), legend.key.size = unit(5.5, 'pt')) + scale_fill_brewer(name="Model", palette="Paired")
save_plot("publication/fig_1.pdf", p1, base_width=4, base_height=3.3)

