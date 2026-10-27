document.addEventListener("DOMContentLoaded", () => {

    const cursor = document.querySelector(".custom-cursor");
    const cursorDot = document.querySelector(".custom-cursor-dot");
    
    document.addEventListener("mousemove", (e) => {
        cursor.style.left = `${e.clientX}px`;
        cursor.style.top = `${e.clientY}px`;
        cursorDot.style.left = `${e.clientX}px`;
        cursorDot.style.top = `${e.clientY}px`;
    });

    const interactiveElements = document.querySelectorAll(".interactive-input, .interactive-btn, a, .trivia-card, .team-card");
    interactiveElements.forEach(el => {
        el.addEventListener("mouseenter", () => cursor.classList.add("active"));
        el.addEventListener("mouseleave", () => cursor.classList.remove("active"));
    });

    initThreeJS();
    loadAnalytics();

    const form = document.getElementById("prediction-form");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        const payload = {};
        formData.forEach((value, key) => {
            payload[key] = isNaN(value) ? value : Number(value);
        });

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if (data.status === "success") {
                updatePredictionUI(data);
                document.getElementById("predictions").scrollIntoView({ behavior: "smooth" });
            }
        } catch (err) {
            console.error("Prediction API Call Failed:", err);
        }
    });
});

function initThreeJS() {
    const container = document.getElementById("threejs-canvas-container");
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const particlesCount = 220;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particlesCount * 3);

    for (let i = 0; i < particlesCount * 3; i++) {
        positions[i] = (Math.random() - 0.5) * 16;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
        color: 0xff9f1c,
        size: 0.055,
        transparent: true,
        opacity: 0.65
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    const nodeGeo = new THREE.IcosahedronGeometry(2.2, 1);
    const nodeMat = new THREE.MeshBasicMaterial({ color: 0xffbf69, wireframe: true, transparent: true, opacity: 0.22 });
    const wireframeNode = new THREE.Mesh(nodeGeo, nodeMat);
    scene.add(wireframeNode);

    camera.position.z = 5.5;

    let mouseX = 0, mouseY = 0;
    document.addEventListener("mousemove", (e) => {
        mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
        mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    function animate() {
        requestAnimationFrame(animate);

        particles.rotation.y += 0.001;
        wireframeNode.rotation.x += 0.0025;
        wireframeNode.rotation.y += 0.004;

        camera.position.x += (mouseX * 0.4 - camera.position.x) * 0.05;
        camera.position.y += (-mouseY * 0.4 - camera.position.y) * 0.05;
        camera.lookAt(scene.position);

        renderer.render(scene, camera);
    }

    animate();

    window.addEventListener("resize", () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

function updatePredictionUI(data) {
    const badge = document.getElementById("prediction-badge");
    badge.innerText = `${data.prediction.toUpperCase()} (${data.risk_level.toUpperCase()})`;
    
    badge.className = "result-badge " + (
        data.prediction === "Slight Injury" ? "badge-slight" :
        data.prediction === "Serious Injury" ? "badge-serious" : "badge-fatal"
    );

    const scores = data.confidence_scores;
    document.getElementById("score-slight").innerText = `${scores["Slight Injury"]}%`;
    document.getElementById("score-serious").innerText = `${scores["Serious Injury"]}%`;
    document.getElementById("score-fatal").innerText = `${scores["Fatal Injury"]}%`;

    document.getElementById("prob-slight").style.width = `${scores["Slight Injury"]}%`;
    document.getElementById("prob-serious").style.width = `${scores["Serious Injury"]}%`;
    document.getElementById("prob-fatal").style.width = `${scores["Fatal Injury"]}%`;
}

Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = 'Rajdhani, sans-serif';

async function loadAnalytics() {
    try {
        const res = await fetch("/api/analytics");
        const data = await res.json();

        if (data.status === "success") {
            const vCtx = document.getElementById("vehicleChart").getContext("2d");
            const vehicleGrad = vCtx.createLinearGradient(0, 0, 400, 0);
            vehicleGrad.addColorStop(0, '#ff9f1c');
            vehicleGrad.addColorStop(1, '#ffbf69');

            new Chart(vCtx, {
                type: 'bar',
                data: {
                    labels: Object.keys(data.vehicle_distribution),
                    datasets: [{
                        label: 'Recorded Accidents',
                        data: Object.values(data.vehicle_distribution),
                        backgroundColor: vehicleGrad,
                        borderRadius: 6,
                        borderSkipped: false
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                        y: { grid: { display: false } }
                    }
                }
            });

            const hCtx = document.getElementById("hourlyChart").getContext("2d");
            const hourlyGrad = hCtx.createLinearGradient(0, 0, 0, 250);
            hourlyGrad.addColorStop(0, 'rgba(255, 159, 28, 0.4)');
            hourlyGrad.addColorStop(1, 'rgba(255, 159, 28, 0.0)');

            new Chart(hCtx, {
                type: 'line',
                data: {
                    labels: Object.keys(data.hourly_distribution).map(h => `${h}:00`),
                    datasets: [{
                        label: 'Accident Density',
                        data: Object.values(data.hourly_distribution),
                        borderColor: '#ff9f1c',
                        borderWidth: 3,
                        pointBackgroundColor: '#ffbf69',
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        fill: true,
                        backgroundColor: hourlyGrad,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }
                    }
                }
            });

            const wCtx = document.getElementById("weatherChart").getContext("2d");
            new Chart(wCtx, {
                type: 'doughnut',
                data: {
                    labels: Object.keys(data.weather_distribution),
                    datasets: [{
                        data: Object.values(data.weather_distribution),
                        backgroundColor: [
                            '#ff9f1c',
                            '#3b82f6',
                            '#10b981',
                            '#8b5cf6',
                            '#f59e0b'
                        ],
                        borderWidth: 2,
                        borderColor: '#0b0d12'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                color: '#e2e8f0',
                                font: { size: 12, weight: '600' }
                            }
                        }
                    },
                    cutout: '68%'
                }
            });
        }
    } catch (e) {
        console.error("Telemetry chart error:", e);
    }
}
