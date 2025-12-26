function getVideoId(url) {
    const match = url.match(/[?&]v=([^&]+)/);
    return match ? match[1] : null;
}

async function getActiveTab() {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true
    });
    return tab
}
  
(async () => {
    const tab = await getActiveTab();
    console.log(tab)
    const videoId = getVideoId(tab.url);

    if (!videoId) {
        document.getElementById("status").innerText = "Not a YouTube video";
        document.body.innerHTML = '<h1 style="text-align: center; margin-top: 50px; color: #e74c3c;">No data available. Please analyze a video first.</h1>';
        
    }
    else{
        const response =await fetch("http://127.0.0.1:5000/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ video_id: videoId })
        });
        const data = await response.json();
        console.log(data)
        document.getElementById('totalComments').textContent = data.totalComments;
        document.getElementById('totalLikes').textContent = data.totalLikes;
        document.getElementById('avgSentiment').textContent = data.avgSentiment;
        document.getElementById('avgLength').textContent = data.avgLength;
        const sentimentSubtext = document.querySelector('.stat-card:nth-child(3) .stat-subtext');
        if (data.avgSentiment > 0.1) {
            sentimentSubtext.textContent = 'Mostly positive 😊';
        } else if (data.avgSentiment < -0.1) {
            sentimentSubtext.textContent = 'Mostly negative 😞';
        } else {
            sentimentSubtext.textContent = 'Mostly neutral 😐';
        }
        const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
        new Chart(sentimentCtx, {
            type: 'doughnut',
            data: {
            labels: ['Positive 😊', 'Neutral 😐', 'Negative 😞'],
            datasets: [{
                data: [
                data.sentimentDist.Positive,
                data.sentimentDist.Neutral,
                data.sentimentDist.Negative
                ],
                backgroundColor: [
                'rgba(46, 204, 113, 0.8)',
                'rgba(52, 152, 219, 0.8)',
                'rgba(231, 76, 60, 0.8)'
                ],
                borderColor: [
                'rgba(46, 204, 113, 1)',
                'rgba(52, 152, 219, 1)',
                'rgba(231, 76, 60, 1)'
                ],
                borderWidth: 3
            }]
            },
            options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                position: 'bottom',
                labels: {
                    color: '#2c3e50',
                    font: { size: 14, weight: '600' },
                    padding: 20
                }
                },
                tooltip: {
                backgroundColor: 'rgba(44, 62, 80, 0.9)',
                titleColor: '#fff',
                bodyColor: '#fff',
                borderColor: '#e8ecf1',
                borderWidth: 1,
                padding: 12,
                callbacks: {
                    label: function(context) {
                    const label = context.label || '';
                    const value = context.parsed;
                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                    const percentage = ((value / total) * 100).toFixed(1);
                    return `${label}: ${value} (${percentage}%)`;
                    }
                }
                }
            }
            }
        });
        document.getElementById('lengthChart').src=data.sentimentTime
        document.getElementById('wordcloudImg').src=data.wordcloud
        }

    })();