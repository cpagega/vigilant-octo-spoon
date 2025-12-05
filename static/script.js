((win, doc) => {
    // create map instance and return map element
    let map = L.map('map').setView([39.5, -98.35], 4);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(map);


    async function fetchData(endpoint, callback) {
        try {
            const response = await fetch("http://localhost:8000" + endpoint);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            callback(data)
        } catch (e) {
            console.error("Error fetching data:", e);
        }
    }


    map.on('click', (e) =>{
        console.log(e.latlng["lat"]);
        console.log(e.latlng["lng"]);
        fetchData(`/prediction/?lat=${e.latlng["lat"]}&lon=${e.latlng["lng"]}`, (data) =>{
            console.log(data)
        });
    });



    // fetchData("/fire", (data) => {
    //     console.log(data.features[0])

    //     // handles a special json for geodata constructed on the python backend
    //     L.geoJSON(data, {
    //         pointToLayer: (feature, latlng) => {
    //             const { frp, scan = 1, track = 1 } = feature.properties;

    //             const dTrack = kmToLat(track);
    //             const dScan = kmToLon(scan, latlng.lat);
  
    //             const bounds = [
    //                 [latlng.lat - dTrack, latlng.lng - dScan],
    //                 [latlng.lat + dTrack, latlng.lng + dScan]
    //             ];


    //             return L.rectangle(bounds, {
    //                 color: "black",
    //                 weight: 1,
    //                 fillOpacity: 0.7,
    //                 fillColor: getColor(frp)
    //             });
    //         },
    //         onEachFeature: onEachFeature
    //     }).addTo(map);
    // });

    function onEachFeature(feature, layer) {
        // does this feature have a property named popupContent?
        if (feature.properties && feature.properties.popupContent) {
            layer.bindPopup(feature.properties.popupContent);
        }
    }

    // frp is probably not the right color metric but using for now to build the map
    function getColor(frp) {
        if (frp < 100) return "#7e0404ff";   // deep dark red
        if (frp < 300) return "#cc0000";     // strong red
        if (frp < 600) return "#f16609ff";   // orange
        if (frp < 900) return "#ffcc00";     // yellow-orange
        if (frp < 1300) return "#ffff66";    // yellow-white
        return "#cce6ff";                    // whitish blue (super hot)
    }

    // convert scan and track from km to degrees
    const kmToLat = km => km / 110.574;
    const kmToLon = (km, lat) => km / (111.320 * Math.cos(lat * Math.PI / 180));

  //  map.addEventListener('zoomend', (e) => {
  //      console.log(e);
  // });

})(window, document);
