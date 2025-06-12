<!DOCTYPE html>
<html>
<head>
    <title>My Python Website</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; background: #f4f4f4; }
        pre { background: #272822; color: #f8f8f2; padding: 15px; border-radius: 8px; overflow-x: auto; }
        img { max-width: 500px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Welcome to My Python Site</h1>
    
    <h2>Python Code Snippet</h2>
    <pre>
import matplotlib.pyplot as plt
import numpy as np

data = np.random.rand(10, 10)
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.savefig('plot.png')
plt.close()
    </pre>

    <h2>Python Generated Heatmap</h2>
    <img src="plot.png" alt="Python Heatmap">

    <p>This heatmap image was created by Python code and shown on this website!</p>
</body>
</html>
