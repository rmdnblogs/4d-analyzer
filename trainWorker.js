importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js');

self.onmessage = async (event) => {
  const { numbers } = event.data;

  // Buat model
  const model = tf.sequential();
  model.add(tf.layers.lstm({ units: 32, inputShape: [10, 4], returnSequences: true }));
  model.add(tf.layers.lstm({ units: 16 }));
  model.add(tf.layers.dense({ units: 4, activation: 'linear' }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  // Normalisasi data
  const normalized = numbers.map(n => {
    const digits = n.toString().padStart(4, '0').split('').map(Number);
    return digits.map(d => d / 9);
  });
  const xs = [], ys = [];
  for (let i = 0; i < normalized.length - 10; i++) {
    xs.push(normalized.slice(i, i + 10));
    ys.push(normalized[i + 10]);
  }

  const xsTensor = tf.tensor3d(xs, [xs.length, 10, 4]);
  const ysTensor = tf.tensor2d(ys, [ys.length, 4]);

  // Latih model
  await model.fit(xsTensor, ysTensor, { epochs: 50, batchSize: 32 });

  // Simpan model ke IndexedDB
  await model.save('indexeddb://4d-model');

  // Bersihkan tensor
  xsTensor.dispose();
  ysTensor.dispose();

  // Kirim pesan ke thread utama
  self.postMessage({ status: 'success' });
};