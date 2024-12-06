const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

const app = express();


const storage = multer.memoryStorage();
const upload = multer({ storage: storage });


let model;
(async () => {
    try {
        model = await tf.loadGraphModel(`file://${path.resolve('model/model.json')}`);
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error.message);
    }
})();


app.get('/', (req, res) => {
    res.send('Welcome to the Express ML API! Use /predict to classify images.');
});


app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        
        if (!req.file) {
            return res.status(400).json({ error: 'No image uploaded' });
        }

        const imageBuffer = req.file.buffer;
        let imageTensor = tf.node.decodeImage(imageBuffer, 3);

        console.log('Image Tensor Shape:', imageTensor.shape);

        imageTensor = tf.image.resizeBilinear(imageTensor, [64, 96]);

        console.log('Resized Image Tensor Shape:', imageTensor.shape);

        imageTensor = imageTensor.div(255.0);

        imageTensor = imageTensor.expandDims(0);

        const predictions = model.predict(imageTensor);

        const output = predictions.arraySync();
        res.json({ predictions: output });
    } catch (error) {
        console.error('Error during prediction:', error.message);
        res.status(500).json({ error: error.message });
    }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
