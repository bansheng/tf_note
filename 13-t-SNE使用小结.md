### 1. Those hyperparameters really matter
+ `perplexity`一般要求在5-50之间，太高太低都不理想，依赖数据密集度
+ `learingrate`根据数据规模来定义，mnist数据建议在0.001，大规模数据适当提高
+ `steps`这个一般为5000

### 2.  Cluster sizes in a t-SNE plot mean nothing
簇大小无意义，显示的效果而已
聚类的相对大小无法在t-sne中显示

### 3. Distances between clusters might not mean anything
t-SNE图中良好分离的簇之间的距离可能没有任何意义。

### 4. Random noise doesn’t always look random.

### 5. You can see some shapes, sometimes

### 6. For topology, you may need more than one plot
单独的低的`perplexity`或者高的`perplexity`可能使图像不准确或者难以理解，可以综合考虑多个`perplexity`
