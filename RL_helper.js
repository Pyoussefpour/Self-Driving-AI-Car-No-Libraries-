function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function derivativesigmoid(x) {
    return x * (1 - x);
}