const csvURL = './../data/iris.csv';
const trainData = tf.data.csv(csvURL, {
    columnConfigs: {
        spacies: {
            isLabel: true
        }
    }
});

const convertedData = trainData.map(({xs, ys}) => {
    const labels = [
        ys.spacies == 'setosa' ? 1 : 0,
        ys.spacies == 'verginica' ? 1 : 0,
        ys.spacies == 'versicolor' ? 1 : 0
    ];
    return { xs: Object.values(xs), ys: Object.values(labels) }
}).batch(10);

console.log(convertedData)