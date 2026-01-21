// ✅ Same-origin API (works locally & on Vercel)
const API = window.location.origin;

/* Collect input data */
function getInputData() {
    return {
        pl_rade: parseFloat(document.getElementById("pl_rade").value),
        pl_bmasse: parseFloat(document.getElementById("pl_bmasse").value),
        pl_eqt: parseFloat(document.getElementById("pl_eqt").value),
        pl_density: parseFloat(document.getElementById("pl_density").value),
        pl_orbper: parseFloat(document.getElementById("pl_orbper").value),
        pl_orbsmax: parseFloat(document.getElementById("pl_orbsmax").value),
        st_luminosity: parseFloat(document.getElementById("st_luminosity").value),
        pl_insol: parseFloat(document.getElementById("pl_insol").value),
        st_teff: parseFloat(document.getElementById("st_teff").value),
        st_mass: parseFloat(document.getElementById("st_mass").value),
        st_rad: parseFloat(document.getElementById("st_rad").value),
        st_met: parseFloat(document.getElementById("st_met").value)
    };
}

/* Prediction */
function predict() {
    fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(getInputData())
    })
    .then(res => res.json())
    .then(data => {
        if (!data || data.error) {
            alert("Prediction failed");
            return;
        }
        document.getElementById("resultTable").classList.remove("hidden");
        document.getElementById("status").innerText = data.label;
        document.getElementById("score").innerText = data.score.toFixed(4);
    })
    .catch(err => {
        console.error(err);
        alert("Prediction failed");
    });
}

/* Show ranking */
function showRanking() {
    fetch(`${API}/ranking`)
    .then(res => res.json())
    .then(data => {
        const table = document.getElementById("rankTable");
        const tbody = table.querySelector("tbody");
        tbody.innerHTML = "";

        if (!Array.isArray(data) || data.length === 0) {
            tbody.innerHTML = "<tr><td colspan='14'>No rankings available yet.</td></tr>";
        } else {
            data.forEach((row, i) => {
                tbody.innerHTML += `
                    <tr>
                        <td>${i + 1}</td>
                        <td>${row.pl_rade}</td>
                        <td>${row.pl_bmasse}</td>
                        <td>${row.pl_eqt}</td>
                        <td>${row.pl_density}</td>
                        <td>${row.pl_orbper}</td>
                        <td>${row.pl_orbsmax}</td>
                        <td>${row.st_luminosity}</td>
                        <td>${row.pl_insol}</td>
                        <td>${row.st_teff}</td>
                        <td>${row.st_mass}</td>
                        <td>${row.st_rad}</td>
                        <td>${row.st_met}</td>
                        <td>${row.score.toFixed(4)}</td>
                    </tr>
                `;
            });
        }

        table.classList.remove("hidden");
        document.getElementById("closeBtn").classList.remove("hidden");
    })
    .catch(err => {
        console.error(err);
        alert("Ranking failed");
    });
}

/* Close ranking */
function closeRanking() {
    document.getElementById("rankTable").classList.add("hidden");
    document.getElementById("closeBtn").classList.add("hidden");
}
