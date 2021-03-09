from gen_entity_link import detect_disambi_result

text = '''
Members of the Black community -- and especially Black women -- are making themselves heard on social media in the aftermath of Meghan and Harry's interview with Oprah.

Many reminded viewers that the racism Meghan faced is a daily reality for the community. Others praised Harry for admitting his own ignorance and privilege regarding racial prejudice, and standing by his wife in the face of public vitriol.

"I’m grateful that Meghan Markle is still here," she added in a series of separate tweets. "We can know racism exists in an institution and still hurt for someone who was hurt by it."

22-year-old Amanda Gorman, who recited her poem at Joe Biden and Kamala Harris' inauguration, tweeted in support of Meghan as well.

In her inaugural poem "The Hill We Climb," Gorman confronted America's racist history and ongoing conflicts head-on.

"We the successors of a country and a time / Where a skinny Black girl / Descended from slaves and raised by a single mother / Can dream of becoming president / Only to find herself reciting for one," she read.

"We are striving to forge a union with purpose / To compose a country committed to all cultures, colors, characters and conditions of man."
'''

def main():
    print(detect_disambi_result(text))
    
if __name__ == '__main__':
    main()