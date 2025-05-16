importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js');

self.onmessage = async (event) => {
  const { numbers } = event.data;

  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: 10000, outputDim: 128, inputLength: 320 }));
  model.add(tf.layers.transformerEncoder({ numHeads: 4, headSize: 32, ffDim: 128, dropout: 0.1 }));
  model.add(tf.layers.globalAveragePooling1d());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 4, activation: 'linear' }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  const normalized = numbers.map(n => {
    const digits = n.toString().padStart(4, '0').split('').map(Number);
    return digits.map(d => d / 9);
  });
  const xs = [], ys = [];
  for (let i = 0; i < normalized.length - 80; i += 3) {
    xs.push(normalized.slice(i, i + 80).flat());
    ys.push(normalized[i + 80]);
  }

  const xsTensor = tf.tensor2d(xs, [xs.length, 320]); // 80 * 4 digits
  const ysTensor = tf.tensor2d(ys, [ys.length, 4]);

  await model.fit(xsTensor, ysTensor, {
    epochs: 100,
    batchSize: 64,
    shuffle: true,
    validationSplit: 0.2
  });

  await model.save('indexeddb://4d-model');
  xsTensor.dispose();
  ysTensor.dispose();

  self.postMessage({ status: 'success' });
};