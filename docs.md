# Tips for bedre konfigurere nettet

1. Ved grunne nett (få lag) er det ofte lurt å velge mellom _sigmoid_ eller eller _hyperbolic tangent_. Det er viktig å merke seg at _sigmoid_ funksjonen har veldig mye dårlige egenskaper når det kommer til dype nett. Hovedsakelig er de dårlige egenskapene hvor lang tid det tar å beregne og hvordan gradientverdiene blir dårligere over tid. Vektene burde konfigureres etter tabellen under.
|function          |lower|upper|
|------------------|:---:|:---:|
|Sigmoid           |    0|    1|
|Hyperbolic tangent|   -1|    1|

2. Ved bruk av dype nett er det lurt å velge RELU (Rectified Linear Unit) eller ELU (Exponential Linear Unit) som aktiveringsfunksjon. RELU og ELU har ikke feilen som Sigmoid har der gradienten blir dårligere og er dermed et brukbart alternativ. Det er veldig vanlig å se en av disse to i dypere nett.

3. Øvre og nedre grense av verdier (se tabellen under) bestemmer ofte hvordan du ønsker at de forskjellige lagene skal bli brukt. Ved å bruke rangen __(0 --> 1)__ ser vi at vektene i dette laget har en mindre effekt på lagene videre nedover i nettet. For å oppnå akkurat det motsatte, at vektene skal ha stor effekt på det som skjer videre nedover i nettet setter vi rangen til __(0 --> 1)__.
 
|lower|upper|Effekt                                        |
|:---:|:---:|:---------------------------------------------|
|    0|    1|Liten effekt på nodene videre nedover i nettet|
|   -1|    1|Stor effekt på nodene videre nedover i nettet |