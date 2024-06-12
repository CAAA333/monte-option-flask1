from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        P0 = float(request.form['P0'])
        K = float(request.form['K'])
        T = float(request.form['T'])
        r = float(request.form['r'])
        sigma = float(request.form['sigma'])
        num_simulations = int(request.form['num_simulations'])
        num_steps = int(request.form['num_steps'])

        dt = T / num_steps
        P = np.zeros((num_simulations, num_steps + 1))
        P[:, 0] = P0

        for t in range(1, num_steps + 1):
            Z = np.random.standard_normal(num_simulations)
            P[:, t] = P[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

        payoff = np.maximum(P[:, -1] - K, 0)
        option_price = np.exp(-r * T) * np.mean(payoff)

        # Plot the simulation results
        plt.figure(figsize=(10, 6))
        plt.plot(P.T, color='grey', alpha=0.1)
        plt.title(f'Monte Carlo Simulation: {num_simulations} Paths of Ticker')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plot_path = os.path.join('static', 'plot.png')
        plt.savefig(plot_path)
        plt.close()

        return render_template('index.html', option_price=f"{option_price:.2f}", plot_url=url_for('static', filename='plot.png'))
    
    return render_template('index.html', option_price=None, plot_url=None)

if __name__ == '__main__':
    app.run(debug=True)
