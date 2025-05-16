importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js');

self.onmessage = async (event) => {
  const { numbers } = event.data;

  // Buat model Transformer sederhana
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: 10000, outputDim: 64, inputLength: 40 }));
  model.add(tf.layers.transformerEncoder({ numHeads: 2, headSize: 32, ffDim: 64, dropout: 0.1 }));
  model.add(tf.layers.globalAveragePooling1d());
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 4, activation: 'linear' }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  // Normalisasi data
  const normalized = numbers.map(n => {
    const digits = n.toString().padStart(4, '0').split('').map(Number);
    return digits.map(d => d / 9);
  });
  const xs = [], ys = [];
  for (let i = 0; i < normalized.length - 30; i += 3) {
    xs.push([...normalized.slice(i, i + 10).flat(), ...normalized.slice(i + 10, i + 20).flat(), ...normalized.slice(i + 20, i + 30).flat()]);
    ys.push(normalized[i + 30]);
  }

  const xsTensor = tf.tensor2d(xs, [xs.length, 120]); // 30 * 4 digits
  const ysTensor = tf.tensor2d(ys, [ys.length, 4]);

  // Latih model
  await model.fit(xsTensor, ysTensor, {
    epochs: 50,
    batchSize: 64,
    shuffle: true
  });

  // Simpan model ke IndexedDB
  await model.save('indexeddb://4d-model');

  // Bersihkan tensor
  xsTensor.dispose();
  ysTensor.dispose();

  self.postMessage({ status: 'success' });
};