import numpy as np
import plotext as plt  

x=[286866,286866,286866,286866,286866,286866,286866,286866,286866,286866,286866,286866,286866,286866,286866,286866,286866,371,371,371,568,568,568,496,496,496,454,454,425,292,225,187,156,150,150,150,149,117,57,57,56,56,56,56,56,56,46,42,35,24,24,15,15,15,15,15,15,15,5,5,4,4,4,2,2,2,1]
result_X = np.log1p(x)

print(f"log1p({x}) = {result_X}")

plt.plot(x, result_X)
plt.xlabel('Original Values')
plt.ylabel('log1p Values')
plt.title('log1p Transformation')
plt.show()  # 在终端中显示ASCII图形