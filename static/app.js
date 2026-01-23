// ===============================
// API BASE (same origin works locally & on Render)
// ===============================
const API = window.location.origin;

// ===============================
// Utility: Safe number parsing
// ===============================
function getNumber(id, fallback = 0) {
    const el = document.getElementById(id);
    if (!el) return fallback;

    const val = el.value;
    const num = parseFloat(val);

    return isNaN(num) ? fallback : num;
}

// ===============================
// Collect input data
// ===============================
function getInputData() {
    return {
        pl_name: document.getElementById("pl_name").value || "Unknown",

        // Planet features
        pl_rade: getNumber("pl_rade"),
        pl_bmasse: getNumber("pl_bmasse"),
        pl_eqt: getNumber("pl_eqt"),
        pl_density: getNumber("pl_density"),
        pl_orbper: getNumber("pl_orbper"),
        pl_orbsmax: getNumber("pl_orbsmax"),

        // Star / orbital features
        st_luminosity: getNumber("st_luminosity"),
        pl_insol: getNumber("pl_insol"),
        st_teff: getNumber("st_teff"),
        st_mass: getNumber("st_mass"),
        st_rad: getNumber("st_rad"),
        st_met: getNumber("st_met")
    };
}

// ===============================
// Predict Habitability
// ===============================
function predict() {
    const payload = getInputData();
    console.log("📤 Sending payload:", payload);

    fetch(`${API}/predict`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    })
    .then(res => {
        if (!res.ok) {
            throw new Error(`HTTP error ${res.status}`);
        }
        return res.json();
    })
    .then(data => {
        console.log("📥 Prediction response:", data);

        if (data.error) {
            alert(`Prediction failed: ${data.error}`);
            return;
        }

        // Show result table
        document.getElementById("resultTable").classList.remove("hidden");
        document.getElementById("status").innerText = data.label;
        document.getElementById("score").innerText = data.score.toFixed(4);
    })
    .catch(err => {
        console.error("❌ Prediction error:", err);
        alert("Prediction failed. Please check inputs and try again.");
    });
}

// ===============================
// Load Rankings
// ===============================
function showRanking() {
    fetch(`${API}/ranking`)
        .then(res => {
            if (!res.ok) {
                throw new Error(`HTTP error ${res.status}`);
            }
            return res.json();
        })
        .then(data => {
            console.log("📊 Ranking data:", data);

            const table = document.getElementById("rankTable");
            const tbody = table.querySelector("tbody");
            tbody.innerHTML = "";

            const rankings = Array.isArray(data.rankings) ? data.rankings : [];

            if (rankings.length === 0) {
                tbody.innerHTML =
                    "<tr><td colspan='4'>No rankings available yet.</td></tr>";
            } else {
                const seen = new Set();
                let rank = 1;

                rankings.forEach(row => {
                    const score = Number(row.confidence_score).toFixed(4);
                    const habitability = row.prediction_value || "N/A";

                    let date = "N/A";
                    if (row.created_at) {
                        const d = new Date(row.created_at);
                        date = `${d.getDate()} ${d.toLocaleString("en-US", {
                            month: "long"
                        })} ${d.getFullYear()}`;
                    }

                    // 🔑 Unique key = DATE + SCORE
                    const key = `${date}-${score}`;

                    if (seen.has(key)) return; // ❌ skip duplicate
                    seen.add(key);

                    tbody.innerHTML += `
                        <tr>
                            <td>${rank++}</td>
                            <td>${score}</td>
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
            console.error("❌ Ranking fetch error:", err);
            alert("Failed to load rankings.");
        });
}


// ===============================
// Close Rankings
// ===============================
function closeRanking() {
    document.getElementById("rankTable").classList.add("hidden");
    document.getElementById("closeBtn").classList.add("hidden");
}
