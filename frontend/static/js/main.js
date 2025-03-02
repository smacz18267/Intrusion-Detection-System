// main.js
document.addEventListener("DOMContentLoaded", function() {
    var socket = io();
    var alertsDiv = document.getElementById("alerts");
    socket.on('new_alert', function(data) {
        var alertItem = document.createElement("div");
        alertItem.className = "alert";
        alertItem.innerHTML = "<strong>" + data.timestamp + "</strong> - " +
                              data.details + " (Score: " + data.anomaly_score.toFixed(2) + ")";
        alertsDiv.prepend(alertItem);
    });
});
