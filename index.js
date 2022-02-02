let doTraining = async (model, xs, ys) => {
    const history = 
        await model.fit(xs, ys, {
            epochs: 500,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log( 'Epoch: ' + epoch + ' Loss: ' + logs.loss )
                }
            }
        })
}

let model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError'
})
model.summary()

let xs = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]);
let ys = tf.tensor2d([-3.0, -1.0, 2.0, 3.0, 5.0, 7.0], [6, 1])

doTraining(model, xs, ys).then((err, res)=>{
    //console.log(model.evaluate(xs, ys));
    alert(model.predict(tf.tensor2d([10],[1, 1])))
});
