import 'babel-polyfill';
import 'bootstrap/dist/css/bootstrap.css'
import * as tf from '@tensorflow/tfjs'
import { MnistData } from './data'

let model

function removeElementsOnInit() {
  document.getElementById('selectTestDataBtn').disabled = true
  document.getElementById('selectTestDataBtn').innerText = 'Model in Training...'

  document.getElementById('log').innerHTML = '';

  var predictionEl = document.getElementById('predictionResult');
  while (predictionEl.firstChild) {
    predictionEl.removeChild(predictionEl.firstChild);
  }
}
removeElementsOnInit()

const createLogEntry = (entry) =>
  document.getElementById('log').innerHTML += '<br>' + entry

// Create Model
const createModel = () => {
  createLogEntry('Create model ...')
  model = tf.sequential()
  createLogEntry('Model created')

  createLogEntry('Add layers ...')
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
  }))

  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }))

  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
  }))

  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }))

  model.add(tf.layers.flatten())

  model.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'VarianceScaling',
    activation: 'softmax'
  }))

  createLogEntry('Layers created')

  createLogEntry('Start compiling ...')
  model.compile({
    optimizer: tf.train.sgd(0.15),
    loss: 'categoricalCrossentropy'
  })
  createLogEntry('Compiled')
}

// Load Data from MNIST
let data
async function load() {
  createLogEntry('Loading MNIST data ...')
  data = new MnistData()
  await data.load()
  createLogEntry('Data loaded successfully')
}

// Train Model
const BATCH_SIZE = 64       // records per batch
const TRAIN_BATCHES = 150   // number batches to train

async function train() {
  createLogEntry('Start Training ...')

  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const batch = tf.tidy(() => {
      const batch = data.nextTrainBatch(BATCH_SIZE)
      batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1])
      return batch
    })

    await model.fit(batch.xs, batch.labels, { batchSize: BATCH_SIZE, epochs: 1 })
    tf.dispose(batch)
    await tf.nextFrame()
  }
  createLogEntry('Training complete')
}

async function main() {
  createModel()
  await load()
  await train()
  document.getElementById('selectTestDataBtn').disabled = false
  document.getElementById('selectTestDataBtn').innerText = 'Predict'
}

async function predict(batch) {
  const div = document.createElement('div')
  showImage(batch, div);
  await makePrediction(batch, div);
}

function showImage(batch, div) {
  tf.tidy(() => {
    div.className = 'prediction-div'
    const image = batch.xs.slice([0, 0], [1, batch.xs.shape[1]])

    const canvas = document.createElement('canvas')
    canvas.className = 'prediction-canvas'
    draw(image.flatten(), canvas)

    const imageLabel = document.createElement('div')
    imageLabel.innerText = 'Image under test:'
    div.appendChild(imageLabel)
    div.appendChild(canvas)
    document.getElementById('predictionResult').prepend(div)
  })
}

async function makePrediction(batch, div) {
  tf.tidy(() => {
    const inputValue = Array.from(batch.labels.argMax(1).dataSync())

    const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]))
    const predictionValue = Array.from(output.argMax(1).dataSync())

    const label = document.createElement('div')
    label.innerHTML = `Original Value: ${inputValue}
                        <br><strong>Prediction Value: ${predictionValue}</strong>`
    if (predictionValue - inputValue == 0) {
      label.innerHTML += `<br>Value recognized <strong>successfully</strong>`
    } else {
      label.innerHTML += `<br>Recognition failed`
    }
    div.appendChild(label)
    // document.getElementById('predictionResult').appendChild(div)
  })
}


// async function predict(batch) {
//   tf.tidy(() => {
//     const inputValue = Array.from(batch.labels.argMax(1).dataSync())

//     const div = document.createElement('div')
//     div.className = 'prediction-div'

//     const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]))
//     const predictionValue = Array.from(output.argMax(1).dataSync())
//     const image = batch.xs.slice([0, 0], [1, batch.xs.shape[1]])

//     const canvas = document.createElement('canvas')
//     canvas.className = 'prediction-canvas'
//     draw(image.flatten(), canvas)

//     const label = document.createElement('div')
//     label.innerHTML = `Original Value: ${inputValue}
//                       <br><strong>Prediction Value: ${predictionValue}</strong>`
//     if (predictionValue - inputValue == 0) {
//       label.innerHTML += `<br>Value recognized <strong>successfully</strong>`
//     } else {
//       label.innerHTML += `<br>Recognition failed`
//     }
//     const imageLabel = document.createElement('div')
//     imageLabel.innerText = 'Image under test:'
//     div.appendChild(imageLabel)
//     div.appendChild(canvas)
//     div.appendChild(label)
//     document.getElementById('predictionResult').appendChild(div)
//   })
// }

function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

document.getElementById('selectTestDataBtn').addEventListener('click', async (el, e) => {
  const batch = data.nextTestBatch(1)
  await predict(batch)
})

main()