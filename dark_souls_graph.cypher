
// Create all locations with their properties
CREATE (firelink:Location {name: 'Firelink Shrine'})
CREATE (asylum:Location {name: 'Northern Undead Asylum', item: 'Peculiar Doll'})
CREATE (catacombs:Location {name: 'The Catacombs'})
CREATE (altar:Location {name: 'Firelink Altar'})
CREATE (newLondo:Location {name: 'New Londo Ruins'})
CREATE (lowerBurg:Location {name: 'Lower Undead Burg'})
CREATE (undeadBurg:Location {name: 'Undead Burg'})
CREATE (parish:Location {name: 'Undead Parish (Bell)'})
CREATE (darkrootGarden:Location {name: 'Darkroot Garden', item: 'Covenant Ring'})
CREATE (sens:Location {name: "Sen's Fortress"})
CREATE (anorLondo:Location {name: 'Anor Londo', item: 'Lordvessel'})
CREATE (dukes:Location {name: "The Duke's Archives", item: 'Broken Pendant'})
CREATE (crystalCave:Location {name: 'Crystal Cave', soul: 'Great Soul'})
CREATE (paintedWorld:Location {name: 'The Painted World of Ariamis'})
CREATE (darkrootBasin:Location {name: 'Darkroot Basin'})
CREATE (oolacileSanctuary:Location {name: 'Oolacile Sanctuary'})
CREATE (royalWood:Location {name: 'Royal Wood'})
CREATE (oolacileTownship:Location {name: 'Oolacile Township'})
CREATE (stoicism:Location {name: 'Battle of Stoicism Gazebo'})
CREATE (chasm:Location {name: 'Chasm of the Abyss'})
CREATE (depths:Location {name: 'The Depths'})
CREATE (blighttown:Location {name: 'Blighttown'})
CREATE (greatHollow:Location {name: 'Great Hollow'})
CREATE (ashLake:Location {name: 'Ash Lake'})
CREATE (valleyDrakes:Location {name: 'Valley of the Drakes'})
CREATE (abyss:Location {name: 'The Abyss', soul: 'Great Soul'})
CREATE (quelaag:Location {name: "Quelaag's Domain (Bell)"})
CREATE (demonRuins:Location {name: 'Demon Ruins'})
CREATE (izalith:Location {name: 'Lost Izalith', soul: 'Great Soul'})
CREATE (kiln:Location {name: 'Kiln of the First Flame'})
CREATE (tombGiants:Location {name: 'Tomb of the Giants', soul: 'Great Soul'})

// Create relationships

// Firelink Shrine connections
CREATE (firelink)-[:CONNECTED {via: 'Peculiar Doll'}]->(asylum)
CREATE (firelink)-[:CONNECTED]->(catacombs)
CREATE (firelink)-[:CONNECTED {via: 'Lordvessel'}]->(altar)
CREATE (firelink)-[:CONNECTED]->(newLondo)
CREATE (firelink)-[:CONNECTED]->(lowerBurg)
CREATE (firelink)-[:CONNECTED]->(undeadBurg)
CREATE (firelink)-[:CONNECTED]->(parish)

// Undead Parish connections
CREATE (parish)-[:CONNECTED]->(undeadBurg)
CREATE (parish)-[:CONNECTED {via: 'Covenant Ring'}]->(darkrootGarden)
CREATE (parish)-[:CONNECTED {via: 'Bells x2'}]->(sens)

// Sen’s Fortress → Anor Londo
CREATE (sens)-[:CONNECTED {via: 'Lordvessel'}]->(anorLondo)

// Anor Londo connections
CREATE (anorLondo)-[:CONNECTED {via: 'Lordvessel'}]->(dukes)
CREATE (anorLondo)-[:CONNECTED {via: 'Peculiar Doll'}]->(paintedWorld)

// Duke’s Archives
CREATE (dukes)-[:CONNECTED {via: 'Broken Pendant'}]->(crystalCave)

// Darkroot Garden connections
CREATE (darkrootGarden)-[:CONNECTED]->(parish)
CREATE (darkrootGarden)-[:CONNECTED]->(darkrootBasin)

// Undead Burg connections
CREATE (undeadBurg)-[:CONNECTED]->(parish)
CREATE (undeadBurg)-[:CONNECTED]->(darkrootBasin)
CREATE (undeadBurg)-[:CONNECTED]->(lowerBurg)

// Darkroot Basin connections
CREATE (darkrootBasin)-[:CONNECTED {via: 'Broken Pendant'}]->(oolacileSanctuary)
CREATE (darkrootBasin)-[:CONNECTED]->(valleyDrakes)

// Oolacile chain
CREATE (oolacileSanctuary)-[:CONNECTED]->(royalWood)
CREATE (royalWood)-[:CONNECTED]->(oolacileTownship)
CREATE (oolacileTownship)-[:CONNECTED]->(stoicism)
CREATE (oolacileTownship)-[:CONNECTED]->(chasm)

// Lower Undead Burg chain
CREATE (lowerBurg)-[:CONNECTED]->(depths)
CREATE (depths)-[:CONNECTED]->(blighttown)
CREATE (blighttown)-[:CONNECTED]->(greatHollow)
CREATE (greatHollow)-[:CONNECTED]->(ashLake)

// New Londo Ruins connections
CREATE (newLondo)-[:CONNECTED]->(valleyDrakes)
CREATE (newLondo)-[:CONNECTED {via: 'Covenant Ring', soul: 'Great Soul'}]->(abyss)

// Valley of the Drakes connections
CREATE (valleyDrakes)-[:CONNECTED]->(blighttown)
CREATE (valleyDrakes)-[:CONNECTED]->(darkrootBasin)

// Blighttown connections
CREATE (blighttown)-[:CONNECTED]->(greatHollow)
CREATE (blighttown)-[:CONNECTED]->(quelaag)
CREATE (quelaag)-[:CONNECTED {via: 'Lordvessel'}]->(demonRuins)
CREATE (demonRuins)-[:CONNECTED {soul: 'Great Soul'}]->(izalith)

// Firelink Altar connections
CREATE (altar)-[:CONNECTED {via: 'Great Souls x4'}]->(kiln)
CREATE (altar)-[:CONNECTED {via: 'Lordvessel', soul: 'Great Soul'}]->(abyss)

// Catacombs connections
CREATE (catacombs)-[:CONNECTED {via: 'Lordvessel', soul: 'Great Soul'}]->(tombGiants)
