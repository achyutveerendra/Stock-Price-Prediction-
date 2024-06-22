document.addEventListener('DOMContentLoaded', function() {
    function updateChart(data) {
        const timestamps = data.map(entry => entry[0]);
        const prices = data.map(entry => entry[1]);

        const trace = {
            x: timestamps,
            y: prices,
            mode: 'lines',
            type: 'scatter',
            name: 'Predicted Stock Prices',
            line: { color: 'rgba(75, 192, 192, 1)' }
        };

        const layout = {
            title: 'Predicted Stock Prices',
            xaxis: {
                title: 'Timestamp',
                type: 'date'
            },
            yaxis: {
                title: 'Price',
                rangemode: 'tozero'
            }
        };

        Plotly.newPlot('chartContainer', [trace], layout);
    }

    function fetchPredictions(ticker, date) {
        fetch(`/predict?ticker=${ticker}&date=${date}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Network response was not ok');
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    displayError(data.error);
                    return;
                }
                updateChart(data);
            })
            .catch(error => {
                console.error('Error fetching predictions:', error);
                displayError(error.message || 'An error occurred while fetching predictions.');
            });
    }

    function displayError(message) {
        const errorMessage = document.getElementById('errorMessage');
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }

    document.getElementById('predictForm').addEventListener('submit', function(event) {
        event.preventDefault();
        const ticker = document.getElementById('ticker').value.trim().toUpperCase();
        const date = document.getElementById('date').value;
        fetchPredictions(ticker, date);
    });
});
