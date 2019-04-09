import 'bootstrap/dist/css/bootstrap.css'
import * as tf from '@tensorflow/tfjs'
// import '@tensorflow/tfjs-node'
// require('@tensorflow/tfjs-node');

// document.getElementById('output').innerHTML = 'JS Works!'

// Y = 2X-1

const model = tf.sequential()
model.add(tf.layers.dense({
  units: 1,
  inputShape: [1]
}))
model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd'
})

const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1])
const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1])

model.fit(xs, ys, { epochs: 150 })
  .then(() => {
    document.getElementById('predictBtn').disabled = false
    document.getElementById('predictBtn').innerText = 'Predict'
    // model.predict(tf.tensor2d([5], [1, 1])).print()
  })

const doPredict = (input) => {
  return model.predict(tf.tensor2d([input], [1, 1]))
}

document.getElementById('predictBtn').addEventListener('click', (el, e) => {
  let input = parseInt(document.getElementById('inputValue').value)
  document.getElementById('output').innerText = doPredict(input)
})