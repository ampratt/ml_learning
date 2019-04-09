import "@babel/polyfill";
import * as tf from "@tensorflow/tfjs"
// import "@tensorflow/tfjs-node"
import iris from "./iris.json"
import irisTesting from "./iris-testing.json"

// convert/setup our data
// const training_X = tf.tensor2d(iris.map(({ species, item }) => Object.values(item)))
const training_X = tf.tensor2d(iris.map(item => [item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]))
const training_y = tf.tensor2d(iris.map(item => [
  item.species === "setosa" ? 1 : 0,
  item.species === "virginica" ? 1 : 0,
  item.species === "versicolor" ? 1 : 0,
]))
const testing_X = tf.tensor2d(irisTesting.map(item => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
]))

// build neural network
const model = tf.sequential()

model.add(tf.layers.dense({
  inputShape: [4],
  activation: "sigmoid",
  units: 5,
}))
model.add(tf.layers.dense({
  inputShape: [5],
  activation: "sigmoid",
  units: 5,
}))
model.add(tf.layers.dense({
  activation: "sigmoid",
  units: 3,
}))
model.add(tf.layers.dense({
  activation: "sigmoid",
  units: 3,
}))
model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(.06),
})
// train/fit our network
console.log('Model Compiled. Start Training now...');
const startTime = Date.now()
model.fit(training_X, training_y, { epochs: 100 })
  .then((history) => {
    const runTime = (Date.now() - startTime) / 1000
    console.log(`DONE in ${runTime} seconds`);
    console.log(history)
    model.predict(testing_X).print()
  })
// test network
