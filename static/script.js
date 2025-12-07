((win, doc) => {
    // create map instance and return map element
    let map = L.map('map').setView([39.5, -98.35], 4);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(map);

    // Get API data - shows a spinner while waiting for API response
    async function fetchData(endpoint, callback) {
        const spinner = doc.getElementById('map-spinner');
        try {
            spinner.classList.remove('d-none'); // Show spinner
            const response = await fetch("http://localhost:8000" + endpoint);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            callback(data);
        } catch (e) {
            console.error("Error fetching data:", e);
        } finally {
            spinner.classList.add('d-none'); // Hide spinner
        }
    }

    // Adds a marker to the map where clicked containing the prediction data
    map.on('click', (e) =>{
        fetchData(`/prediction?lat=${e.latlng["lat"]}&lon=${e.latlng["lng"]}`, (data) =>{
        // handles a special json for geodata constructed on the python backend
        L.geoJSON(data, {
            pointToLayer: (feature, latlng) => {
                const color = getColor(feature.properties.label);
                return L.circleMarker(latlng, {
                    radius: 8,
                    color: "#000",
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8,
                    fillColor: color
                });
            },
            onEachFeature: onEachFeature
        }).addTo(map);
        });
    });



    // Adds a pop up to the marker
    function onEachFeature(feature, layer) {
        // does this feature have a property named popupContent?
        if (feature.properties && feature.properties.popupContent) {
            layer.bindPopup(feature.properties.popupContent);
        }
    }

    // Sets the marker color
    function getColor(label) {
        switch(label) {
            case 'low_risk': return "#00ff00";
            case 'medium_risk': return "#ff8800ff";
            case 'high_risk': return "#ff0000";
        }
    }

})(window, document);
