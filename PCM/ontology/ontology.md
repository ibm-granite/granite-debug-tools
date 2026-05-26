# PCM Ontology

Hybrid ontology for personal knowledge graph extraction from conversational text.

## Design Principles

1. **Standard-first**: reuse standard vocabularies wherever possible
2. **Custom only when necessary**: define `pcm:` classes/properties only when no standard covers the concept
3. **Competency-question driven**: additions validated against 500 LongMemEval benchmark questions (see `competency_analysis.md`)

## Namespaces

| Prefix | URI | Source |
|--------|-----|--------|
| `schema:` | `http://schema.org/` | Schema.org |
| `ob:` | `https://w3id.org/ontobio#` | OntoBio (Dutta & Arzoo, 2024) |
| `foaf:` | `http://xmlns.com/foaf/0.1/` | FOAF |
| `prov:` | `http://www.w3.org/ns/prov#` | W3C PROV-O |
| `skos:` | `http://www.w3.org/2004/02/skos/core#` | W3C SKOS |
| `rel:` | `http://purl.org/vocab/relationship/` | Relationship vocab |
| `bio:` | `http://purl.org/vocab/bio/0.1/` | BIO vocab |
| `cv:` | `http://rdfs.org/resume-rdf/cv.rdfs#` | ResumeRDF |
| `opencare:` | `https://w3id.org/opencare#` | OpenCare |
| `sioc:` | `http://rdfs.org/sioc/ns#` | SIOC (Semantically-Interlinked Online Communities) |
| `pcm:` | `https://w3id.org/pcm/` | **Custom** (Personal Conversational Memory) |
| `rdfs:` | `http://www.w3.org/2000/01/rdf-schema#` | RDF Schema |
| `xsd:` | `http://www.w3.org/2001/XMLSchema#` | XML Schema |

## What comes from where

### Schema.org
- **Person & identity**: `schema:Person`, `schema:name`, `schema:gender`, `schema:knowsLanguage`
- **Actions** (full taxonomy): `BuyAction`, `PayAction`, `RepairAction`, `TravelAction`, `CreateAction`, `UpdateAction`, `DeleteAction`, `ReadAction`, `WriteAction`, `SearchAction`, `CommunicateAction`, `AchieveAction`, `AssessAction`, `ConsumeAction`, `InteractAction`, `OrganizeAction`, `PlayAction`, `TradeAction`, `TransferAction`, `UseAction`, `MoveAction`, `ReviewAction`
- **Action properties**: `agent`, `object`, `result`, `location`, `instrument`, `participant`, `startDate`, `endDate`, `actionStatus`
- **Events**: `schema:Event`, `schema:SocialEvent`
- **Things**: `schema:Product`, `schema:Vehicle`, `schema:CreativeWork`
- **Product properties**: `brand`, `model`, `color`, `price`, `priceCurrency`
- **Ownership**: `schema:owns`
- **Places**: `schema:Place`, `City`, `Country`, `State`, `Continent`, `TouristAttraction`
- **Organizations**: `schema:Organization`, `EducationalOrganization`, `GovernmentOrganization`, `MedicalOrganization`
- **Work**: `schema:hasOccupation`, `schema:Occupation`, `schema:jobTitle`, `schema:affiliation`, `schema:memberOf`
- **Health**: `schema:MedicalCondition`, `schema:associatedAnatomy`

### OntoBio (`ob:`)
- **Family relationships**: full hierarchy with property chains — `hasFather`, `hasMother`, `hasSibling`, `hasSpouse`, `hasUncle`, `hasCousin`, `hasInLaw`, etc. (30+ properties)
- **Travel**: `ob:Travel`, `travelFrom`, `travelTo`, `reasonForTravel`, `modeOfTransport`, `travelYear`
- **Habits**: `ob:Habit`, `DailyRoutine`, `FoodHabit`, `DressingHabit`, `hasHabit`, `hasRoutine`, `hasFoodHabit`
- **Activities**: `ob:Activity`, `PhysicalActivity`, `CreativeActivity`, `RecreationalActivity`, `DailyLivingActivity`, `OccupationalActivity`, `ReligiousActivity`
- **Food & meals**: `ob:Meal`, `Breakfast`, `Lunch`, `Dinner`, `FoodItem`, `EdibleFood`, `DrinkableFood`, `hasMeal`, `hasFoodItem`
- **Traits**: `ob:Trait`, `PersonalityTrait`, `PhysicalAppearance`, `PhysicalTrait`, `hasHeight`, `hasWeight`, `hasEyeColor`
- **Education**: `BachelorsDegree`, `MastersDegree`, `DoctoralDegree`, `hasDegree`, `degreeAwardedBy`
- **Work**: `ob:employedAt`, `hasWorkPlace`, `tenureFrom`, `tenureTill`
- **Residence**: `ob:Residence`, `residedIn`, `residedFrom`, `residedTill`
- **Causality**: `ob:hasCausality`, `causalContext`, `followedBy`, `precededBy`
- **Pets**: via Wikidata classes (`wd:Q39201`), `ob:isPetOf`
- **Awards**: `ob:hasAward`, `awardedTo`, `conferredBy`
- **Goals**: `ob:hasGoal`
- **Religion**: `ob:followsReligion`, `ob:Religion`
- **Identity**: `ob:hasEthnicity`, `ob:Man`, `ob:Woman`

### Relationship Vocab (`rel:`)
- Social relationships: `friendOf`, `closeFriendOf`, `acquaintanceOf`, `colleagueOf`, `worksWith`, `neighborOf`, `mentorOf`, `apprenticeTo`, `engagedTo`

### OpenCare
- Health records: `opencare:HealthRecord`, `Disease`, `Symptom`, `TreatmentProcedure`, `Surgery`, `DentalProcedure`
- Health properties: `diagnosedWith`, `hasSymptom`, `hasTreatmentProcedure`, `encounterDate`
- `ob:treatedBy` (OntoBio extension of OpenCare)

### BIO Vocab
- `bio:Employment`, `bio:Marriage`

### CV / ResumeRDF
- `cv:Education`, `cv:WorkHistory`, `cv:hasWorkHistory`

### FOAF
- `foaf:Person`, `foaf:name`, `foaf:firstName`, `foaf:family_name`, `foaf:nick`, `foaf:interest`

### PROV-O
- `prov:wasDerivedFrom`, `prov:generatedAtTime`, `prov:wasAttributedTo`

### SKOS
- `skos:Concept`, `skos:ConceptScheme`, `skos:prefLabel`, `skos:altLabel`, `skos:definition`, `skos:broader`, `skos:narrower`, `skos:related`

## PCM Custom Extensions (`pcm:`)

These are concepts **not covered by any standard ontology** we found:

### pcm:Preference
A personal preference (e.g. "I prefer Italian food", "I like running in the morning").

| Term | Type | Description |
|------|------|-------------|
| `pcm:Preference` | Class | A personal preference |
| `pcm:hasPreference` | ObjectProperty | Person -> Preference |
| `pcm:preferenceType` | DatatypeProperty | Category (food, music, brand, etc.) |
| `pcm:preferenceValue` | DatatypeProperty | The preference value |

### pcm:ProblemEvent
A problem or issue that occurred (e.g. "my GPS broke", "car wouldn't start").

| Term | Type | Description |
|------|------|-------------|
| `pcm:ProblemEvent` | Class (subclass of schema:Event) | A problem or issue |
| `pcm:problemType` | DatatypeProperty | Category (mechanical, software, etc.) |
| `pcm:affects` | ObjectProperty | What entity is impacted |

### pcm:ServiceEvent
A service interaction (e.g. "took car for oil change", "called plumber").

| Term | Type | Description |
|------|------|-------------|
| `pcm:ServiceEvent` | Class (subclass of schema:Event) | A service interaction |
| `pcm:serviceType` | DatatypeProperty | Category (repair, maintenance, etc.) |

### Schedule & Frequency
Uses `schema:Schedule` with ISO 8601 durations.

| Term | Type | Description |
|------|------|-------------|
| `schema:Schedule` | Class | A recurrence pattern for events or activities |
| `schema:eventSchedule` | Property | Links Event/Habit/Activity to a Schedule |
| `schema:repeatFrequency` | Property | ISO 8601 duration: "P1D"=daily, "P1W"=weekly, "P1M"=monthly |
| `schema:repeatCount` | Property | Times per period (e.g. 3 for "three times a week") |
| `schema:byDay` | Property | Day(s) of the week |
| `schema:byMonth` | Property | Month(s) of the year (1-12) |
| `schema:duration` | Property | Duration of each occurrence |

### pcm:Pet Domestic animals. Schema.org has no Animal/Pet class.

| Term | Type | Description |
|------|------|-------------|
| `pcm:Pet` | Class (subclass of foaf:Agent) | A domestic animal |
| `pcm:hasPet` | ObjectProperty | Person → Pet |
| `pcm:petSpecies` | DatatypeProperty | dog, cat, parrot, fish, etc. |
| `pcm:petBreed` | DatatypeProperty | Breed, if applicable |

### pcm:SocialMediaAccount Social media accounts. Extends SIOC's UserAccount.

| Term | Type | Description |
|------|------|-------------|
| `pcm:SocialMediaAccount` | Class (subclass of sioc:UserAccount) | A social media account |
| `pcm:hasSocialAccount` | ObjectProperty | Person → SocialMediaAccount |
| `pcm:platform` | DatatypeProperty | Instagram, TikTok, YouTube, etc. |
| `pcm:followerCount` | DatatypeProperty | Number of followers (integer) |
| `schema:identifier` | Property | Handle / @username |

### pcm:Clothing Wearable garments. Extends schema:Product.

| Term | Type | Description |
|------|------|-------------|
| `pcm:Clothing` | Class (subclass of schema:Product) | A garment or accessory |
| `pcm:wears` | ObjectProperty | Person → Clothing |
| `pcm:clothingType` | DatatypeProperty | shoes, shirt, jacket, dress, etc. |

### pcm:Plant Plants grown or maintained by a person.

| Term | Type | Description |
|------|------|-------------|
| `pcm:Plant` | Class | A plant |
| `pcm:grows` | ObjectProperty | Person → Plant |
| `pcm:plantSpecies` | DatatypeProperty | tomato, basil, marigold, etc. |

### pcm:Collection Personal collections of items.

| Term | Type | Description |
|------|------|-------------|
| `pcm:Collection` | Class | A personal collection |
| `pcm:hasCollection` | ObjectProperty | Person → Collection |
| `pcm:collectionType` | DatatypeProperty | coins, postcards, stamps, etc. |
| `pcm:collectionSize` | DatatypeProperty | Number of items (integer) |

## Remaining Gaps & Open Questions

1. **Preference vs. Habit boundary** — Some preferences are habits ("I always buy Shell gas") and some habits imply preferences. The boundary is fuzzy. May need clearer modeling guidelines.

2. **Spending/financial tracking** — 50 competency questions about money, but only `schema:price` exists. Consider `pcm:amountSpent` on `schema:BuyAction` or a `pcm:Expense` class.

3. **Quantity/aggregation** — 168 competency questions ask "how many" — the KG must have countable, typed instances for SPARQL COUNT queries to work.

## References

- **OntoBio**: Dutta, B. & Arzoo, S. (2026). "Towards a Biographical Ontology: The OntoBio Framework and Its Applications." Knowledge Organization, 53(1). https://w3id.org/ontobio
- **Schema.org**: https://schema.org
- **FOAF**: http://xmlns.com/foaf/spec/
- **PROV-O**: https://www.w3.org/TR/prov-o/
- **SKOS**: https://www.w3.org/TR/skos-reference/
- **Relationship vocab**: http://purl.org/vocab/relationship/
- **OpenCare**: https://w3id.org/opencare
