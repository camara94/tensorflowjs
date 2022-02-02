let doTraining = async (model) => {
     
}
let model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError'
})
//model.summary()