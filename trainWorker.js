importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js');

self.onmessage = async (event) => {
  const { numbers, minData } = event.data;

  const windowSize = Math.min(minData, numbers.length); // Sesuaikan jendela dengan data
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: 10000, outputDim: 64, inputLength: windowSize * 4 }));
  model.add(tf.layers.transformerEncoder({ numHeads: 2, headSize: 16, ffDim: 64, dropout: 0.1 }));
  model.add(tf.layers.globalAveragePooling1d());
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.1 }));
  model.add(tf.layers.dense({ units: 4, activation: 'linear' }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  const normalized = numbers.map(n => {
    const digits = n.toString().padStart(4, '0').split('').map(Number);
    return digits.map(d => d / 9);
  });
  const xs = [], ys = [];
  for (let i = 0; i < normalized.length - windowSize; i += 3) {
    xs.push(normalized.slice(i, i + windowSize).flat());
    ys.push(normalized[i + windowSize]);
  }

  const xsTensor = tf.tensor2d(xs, [xs.length, windowSize * 4]);
  const ysTensor = tf.tensor2d(ys, [ys.length, 4]);

  await model.fit(xsTensor, ysTensor, {
    epochs: 50, // Kurangi epoch untuk data kecil
    batchSize: 32,
    shuffle: true,
    validationSplit: 0.2
  });

  await model.save('indexeddb://4d-model');
  xsTensor.dispose();
  ysTensor.dispose();

  self.postMessage({ status: 'success' });
};