class DQN {
    constructor(layerSizes) {
        this.sizes   = layerSizes.slice();
        this.levels  = [];        // store w, b, actionvations, preactivations, and deltas [{w,b,a,z,δ}, …]   one per layer (except input)

        for (let l = 0; l < layerSizes.length - 1; l++) {
            const nIn  = layerSizes[l];
            const nOut = layerSizes[l + 1];
            const limit = Math.sqrt(6 / (nIn + nOut));   // Xavier/Glorot uniform for better stability

            const w = Array.from({ length: nIn  },
                      () => Array.from({ length: nOut },
                      () => (Math.random() * 2 - 1) * limit));

            const b = Array.from({ length: nOut },
                      () => 0);

            this.levels.push({ w, b, a: null, z: null, delta: null });
        }
    }

    /* ---------- forward pass (ReLU hidden, linear output) ---------- */
    forward(input) {
        console.log(input);
        let aPrev = input.slice();                       // copy
        for (const lvl of this.levels) {
            const { w, b } = lvl;                       // pull out w is the weights, b is the biases
            const z = new Array(b.length);

            for (let j = 0; j < b.length; j++) {
                let s = b[j];
                for (let i = 0; i < w.length; i++) s += aPrev[i] * w[i][j];             // {} is optional, since there is only one statment
                z[j] = s;
            }
            const a = lvl === this.levels.at(-1)           // apply ReLU activation to every layer except the output layer
                    ? z.slice()
                    : z.map(relu);

            lvl.z = z;
            lvl.a = a;
            aPrev = a;
        }
        return aPrev;                                     // network output
    }

    compute_gradient(obs, Q_target, gamma = 0.99){
        // calculate the loss from target
        const yPred = this.forward(obs.state.sensors);
        const Q_next = Q_target.forward(obs.next_state.sensors);
        const max_Q_next = Math.max(...Q_next);
        let target = obs.done ? obs.reward : obs.reward + gamma * max_Q_next;
        let td_error = yPred[obs.action] - target;
        let loss = 0.5 * (td_error ** 2);
        let delta = yPred.map((q,j) => j === obs.action ? td_error : 0);

        // Init the gradient
        const grad = this.levels.map(lvl => ({
            dw: Array.from({ length: lvl.w.length },
                 () => new Array(lvl.w[0].length).fill(0)),
            db: new Array(lvl.b.length).fill(0)
        }));

        
        for (let layer = this.levels.length - 1; layer >= 0; layer--) {
            const lvl = this.levels[layer];
            const prevA = layer === 0 ? obs.state : this.levels[layer - 1].a;

            for (let j = 0; j < lvl.b.length; j++) {
                grad[layer].db[j] += delta[j];
                for (let i = 0; i < prevA.length; i++)
                    grad[layer].dw[i][j] += prevA[i] * delta[j];
            }

            if (layer > 0) {
                const prevZ = this.levels[layer - 1].z;
                const deltaPrev = new Array(prevZ.length).fill(0);

                for (let i = 0; i < prevZ.length; i++) {
                    let sum = 0;
                    for (let j = 0; j < delta.length; j++)
                        sum += this.levels[layer].w[i][j] * delta[j];
                    deltaPrev[i] = sum * reluPrime(prevZ[i]);
                }
                delta = deltaPrev;
            }
        }

        return grad;

    }

    makeEmptyGrad() {
        return this.levels.map(lvl => ({
            dw: lvl.w.map(row => row.map(() => 0)),
            db: lvl.b.map(() => 0)
        }));
    }


    static add_gradient(grad1, grad2) {
        for (let l = 0; l < grad1.length; l++) {
            const dw1 = grad1[l].dw;
            const db1 = grad1[l].db;
            const dw2 = grad2[l].dw;
            const db2 = grad2[l].db;

            for (let j = 0; j < dw1.length; j++) {
                for (let i = 0; i < dw1[0].length; i++) {
                    dw1[j][i] += dw2[j][i];
                }
            }
            for (let i = 0; i < db1.length; i++) {
                db1[i] += db2[i];
            }
        }
        return grad1;
    }

        /* ---------- utility: scale (or divide) a gradient ---------- */
    static divideGrad(grad, factor) {
        // multiply every entry of g in-place by `factor`
        let divisor = 1 / factor;
        for (let l = 0; l < grad.length; l++) {
            const dw = grad[l].dw, db = grad[l].db;

            // weights
            for (let i = 0; i < dw.length; i++)
                for (let j = 0; j < dw[0].length; j++)
                    dw[i][j] *= divisor;

            // biases
            for (let j = 0; j < db.length; j++)
                db[j] *= divisor;
        }
        return grad;
    }
    

    updateParams(gradAvg, lr) {
        for (let l = 0; l < this.levels.length; l++) {
            const lvl = this.levels[l];
            for (let j = 0; j < lvl.b.length; j++) {
                lvl.b[j] -= lr * gradAvg[l].db[j];
                for (let i = 0; i < lvl.w.length; i++)
                    lvl.w[i][j] -= lr * gradAvg[l].dw[i][j];
            }
        }
    }

    /* ---------- Polyak soft update:  θ_target ← τ θ_src + (1−τ) θ_target ---------- */
    softUpdateFrom(sourceNet, tau = 0.005) {
        for (let l = 0; l < this.levels.length; l++) {
            const tgt = this.levels[l],
                  src = sourceNet.levels[l];

            for (let j = 0; j < tgt.b.length; j++)
                tgt.b[j] = tau * src.b[j] + (1 - tau) * tgt.b[j];

            for (let i = 0; i < tgt.w.length; i++)
                for (let j = 0; j < tgt.w[0].length; j++)
                    tgt.w[i][j] = tau * src.w[i][j] + (1 - tau) * tgt.w[i][j];
        }
    }

    clone() {                        // deep copy (for target network)
        const net = new DQN(this.sizes);
        for (let l = 0; l < this.levels.length; l++) {
            const src = this.levels[l], dst = net.levels[l];
            dst.b = src.b.slice();
            for (let i = 0; i < src.w.length; i++)
                dst.w[i] = src.w[i].slice();
        }
        return net;
    } 

    /* ---------- parameter I/O ---------- */
    getParameters() {
        /* deep-copy every weight & bias */
        return {
            sizes: this.sizes.slice(),
            levels: this.levels.map(lvl => ({
                w: lvl.w.map(row => row.slice()),
                b: lvl.b.slice()
            }))
        };
    }

    setParameters(paramObj) {
        if (!paramObj
            || !Array.isArray(paramObj.sizes)
            || paramObj.sizes.length !== this.sizes.length
            || !paramObj.sizes.every((v, i) => v === this.sizes[i])) {
            throw new Error("setParameters fcn, network architecture mismatch");
        }

        for (let l = 0; l < this.levels.length; l++) {
            const src = paramObj.levels[l], dst = this.levels[l];

            if (src.w.length !== dst.w.length ||
                src.w[0].length !== dst.w[0].length ||
                src.b.length !== dst.b.length)
                throw new Error(`setParameters: shape mismatch at layer ${l}`);

            dst.b = src.b.slice();
            for (let i = 0; i < dst.w.length; i++)
                dst.w[i] = src.w[i].slice();
        }
    }
}

/* ---------- helpers ---------- */
const relu       = x => Math.max(0, x);
const reluPrime  = x => x > 0 ? 1 : 0;


// const policy = new DQN([5,16,4]);
// const target = policy.clone();
// // let output = policy.forward([1,2,3,4,5]);
// // let grad = policy.compute_gradient([1,2,3,4,5], target);
// // console.log(grad);
// // console.log(output);
// // // console.log(policy.levels);

// const dummyObs = {
//     state:       [1, 2, 3, 4, 5],  // current state (array, length 5)
//     next_state:  [2, 3, 4, 5, 6],  // next state (same length)
//     action:      0,                // action index 0‥3
//     reward:      1.0,              // scalar reward
//     done:        false             // episode-done flag
// };
// let output = policy.forward(dummyObs.state);
// console.log(output);
// const grad = policy.compute_gradient(dummyObs, target);
// // console.log(grad);
