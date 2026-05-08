import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  //primeira cadad da rede:
  //entrada de 7 posições ( idade normalizada + 3 cores + 3 localizações)

  //80 neuronios = aqui coloquei isso pq tem pouca base de treino
  //quanto mais neuronios, masi complexidade a rede pode aprender, consequentemente, mais processamento vai ser usado

  //A reLU age como filtro:
  // é como se ela deixasse somente os dados interessantes seguirem viagem na rede
  // se a informação chegou nesse neuronio é positiva, passa para frente.
  //se for zero ou negativa, pode jogar fora, nao vai servir para nada
  model.add(tf.layers.dense({ inputShape: [7], units: 80, activation:'relu' }));

  //Saída: 3 neuronios
  // Pq? pois no momento possuo 3 categorias ( premium, medium e basic)

  //activation: softmax normaliza a saida em probabilidades
  model.add(tf.layers.dense({units:3,activation:'softmax'}))

  // compilando o modelo
  // optimizer adam (adaptive moment estimation)
  //é um treinador pessoal moderno para redes neurais:
  //ajusta os pesos de forma eficiente e inteligente

  //loss: categoricalCrossentropy
  // ele compara o que o modelo "acha" ( os scores de cada categoria) com a resposta certa
  // a categoria premium será sempre [1,0,0]   de acordo com o tensorLabels

  // quanto mais distante da previsao do modelo da resposta correta
  // maior o erro (loss)
  // Exemplo classico: classificação de imagens, recomendação e categorização de usuário
  // qualquer coisa em que a resposta certa é apenas uma entre várias possíveis
  model.compile({optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy']})

}
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

const model = trainModel(inputXs, outputYs);
