
library(M3C)
library(ComplexHeatmap)
library(data.table)
library(ggplot2)

setwd("M3C")

dir.create("Figures")
dir.create("ClusterLables")

consensus_object <- "AE_pancancer.rds"
cdf_plot <- "cdf_plot.pdf"
pac_plot <- "pac_plot.pdf"
pvalue_plot <- "pvalue_plot.pdf"
rsci_plot <- "rsci_plot.pdf"
scores_path <- "res_scores.csv"

ip_path <- "AE_reduced_dataset.csv"
mydata <- fread(ip_path, header = TRUE, data.table = FALSE, stringsAsFactors = FALSE)
mydata[1:5, 1:5]
rownames(mydata) <- mydata[,1]
mydata <- as.data.frame(t(mydata[,-1]))
mydata[1:5, 1:5]

maxK <- 20

#=====================================================================================================
# Data frame or matrix: Contains the data, with samples as columns and rows as features
res.ConClust <- M3C(mydata, cores = 14, iters = 100, maxK = maxK, pItem = 0.8,
                    des = NULL, ref_method = "reverse-pca", repsref = 100,
                    repsreal = 1000, clusteralg = "km",
                    pacx1 = 0.1, pacx2 = 0.9, seed = 2111133, objective = "PAC",
                    removeplots = TRUE, silent = FALSE, fsize = 18, method = 1,
                    lambdadefault = 0.1, tunelambda = TRUE, lseq = seq(0.02, 0.1, by = 0.01), 
                    lthick = 0.5, dotsize = 1)

saveRDS(res.ConClust, file = consensus_object) 

res_scores <- res.ConClust$scores
fwrite(res_scores, scores_path, row.names = FALSE, col.names = TRUE)

ggsave(cdf_plot, res.ConClust$plots[[1]])
ggsave(pac_plot, res.ConClust$plots[[2]])
ggsave(pvalue_plot, res.ConClust$plots[[3]])
ggsave(rsci_plot, res.ConClust$plots[[4]])



x <- c("#A6CEE3","#1F78B4","#B2DF8A","#33A02C","#FB9A99","#E31A1C","#FDBF6F","#FF7F00","#CAB2D6","#6A3D9A","#FFFF99","#B15928",
       "#bd18ea", #magenta
       "#2ef4ca", #aqua
       "#f4cced", #pink,
       "#f4cc03", #lightorange
       "#05188a", #navy,
       "#e5a25a", #light brown
       "#06f106", #bright green
       "#85848f", #med gray
       "#000000", #black
       "#076f25", #dark green
       "#93cd7f",#lime green
       "#4d0776", #dark purple
       "#ffffff" #white
)
names(x) <- as.character(seq(1, length(x), by=1))

for (i in seq(2, maxK)){
  
  # get cc matrix and labels
  ccmatrix <- res.ConClust$realdataresults[[i]]$consensus_matrix
  annon <- res.ConClust$realdataresults[[i]]$ordered_annotation
  fwrite(annon, paste0("ClusterLables/Class_Consensus_Clustering_K_", i, ".csv"), row.names = TRUE, col.names = TRUE)
  
  # do heatmap
  n <- 10
  seq <- rev(seq(0,255,by=255/(n)))
  palRGB <- cbind(seq,seq,255)
  mypal <- rgb(palRGB,maxColorValue=255)
  ha = HeatmapAnnotation(df= data.frame(Cluster=as.character(annon[,1])), col = list(Cluster=x))
  
  pdf(paste0("Figures/Class_Consensus_Clustering_K_", i, "_HeatMap.pdf"), width = 8, height = 7)
  draw(Heatmap(ccmatrix, name = "Consensus_index", top_annotation = ha,
               col=mypal, show_row_dend = FALSE, use_raster = TRUE, 
               show_column_dend = FALSE, cluster_rows = FALSE, cluster_columns = FALSE,
               show_column_names = FALSE))
  dev.off()

}

warnings()

