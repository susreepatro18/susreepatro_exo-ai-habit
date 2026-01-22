// ✅ Same-origin API (works locally & on Vercel)
const API = window.location.origin;

/* Collect input data */
function getInputData() {
    return {
        pl_name: document.getElementById("pl_name").value || "Unknown",
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
        console.log("Ranking data:", data);
        const table = document.getElementById("rankTable");
        const tbody = table.querySelector("tbody");
        tbody.innerHTML = "";

        // Handle the response object with rankings array
        let rankings = data.rankings || [];
        
        if (!Array.isArray(rankings) || rankings.length === 0) {
            tbody.innerHTML = "<tr><td colspan='4'>No rankings available yet.</td></tr>";
            console.log("No rankings found");
        } else {
            // Remove duplicates - keep only unique confidence scores
            const seenScores = new Set();
            const deduplicated = [];
            
            rankings.forEach((row) => {
                const score = parseFloat(row.confidence_score).toFixed(4);
                if (!seenScores.has(score)) {
                    seenScores.add(score);
                    deduplicated.push(row);
                }
            });
            
            console.log(`Displaying ${deduplicated.length} unique rankings (deduplicated from ${rankings.length})`);
            deduplicated.forEach((row, i) => {
                const habitability = row.prediction_value || "N/A";
                const score = row.confidence_score || 0;
                
                // Format date as "date month year" (e.g., "22 January 2026")
                let date = "N/A";
                if (row.created_at) {
                    const dateObj = new Date(row.created_at);
                    const day = dateObj.getDate();
                    const month = dateObj.toLocaleString('en-US', { month: 'long' });
                    const year = dateObj.getFullYear();
                    date = `${day} ${month} ${year}`;
                }
                
                tbody.innerHTML += `
                    <tr>
                        <td>${i + 1}</td>
                        <td>${score.toFixed(4)}</td>
                        <td>${habitability}</td>
                        <td>${date}</td>
                    </tr>
                `;
            });
        }

        table.classList.remove("hidden");
        document.getElementById("closeBtn").classList.remove("hidden");
    })
    .catch(err => {
        console.error("Ranking fetch error:", err);
        alert("Ranking failed: " + err.message);
    });
}

/* Close ranking */
function closeRanking() {
    document.getElementById("rankTable").classList.add("hidden");
    document.getElementById("closeBtn").classList.add("hidden");
}
