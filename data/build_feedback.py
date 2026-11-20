"""Generate grouped, original retail feedback examples without customer data."""
import json
from pathlib import Path

SCENARIOS = {
    "product_issue": (
        ("the wireless mouse keeps disconnecting", "I swapped batteries and it still drops out."),
        ("the glass kettle leaks around the handle", "There's water under it after every use."),
        ("the zipper on the daypack came apart", "Only used it for the morning commute."),
        ("the striped shirt shrank after one wash", "I followed the care instructions."),
        ("the desk lamp flickers even with a fresh bulb", "It is distracting at night."),
        ("one earbud stopped playing sound", "Charging the case overnight didn't help."),
        ("the door on the countertop oven won't close", "The latch feels loose."),
        ("the storage bin cracked with hardly anything in it", "It's a shame because the size was perfect."),
        ("the moisturizer arrived sealed but smelled rancid", "I did not want to use it."),
        ("the travel mug lets coffee seep through the lid", "It was upright in the cup holder."),
        ("the sewing kit was missing its needles", "The packaging itself looked fine."),
        ("the baking tray warped the first time I used it", "The price was good, but I can't use it now."),
    ),
    "staff_praise": (
        ("the cashier helped my dad read the payment screen", "They were patient the entire time."),
        ("an employee fetched a basket before I even asked", "Small thing but it made my visit easier."),
        ("the returns clerk went out of their way to explain my options", "I left feeling heard."),
        ("the person at the shoe counter found a better fit for me", "No pressure to buy the more expensive pair."),
        ("the floor assistant walked me to the right shelf", "I had been going in circles before that."),
        ("the support person stayed late to help with my pickup", "Please pass on my thanks."),
        ("the cashier spotted the ripped bag and packed everything again", "I appreciated how kind they were."),
        ("the associate at the service desk kept the kids entertained while I waited", "What a lovely person."),
        ("the sales assistant explained which charger I actually needed", "That saved me a second trip."),
        ("the team at the register handled a mix-up without blaming anyone", "Really professional."),
        ("a staff member carried the heavy box out to the car", "I would've struggled on my own."),
        ("the employee dealing with the return was brilliant", "The product disappointed me, but their help deserves credit."),
    ),
    "pricing": (
        ("the advertised discount vanished at checkout", "It made the total more than I planned to spend."),
        ("the receipt charged full price for the clearance towels", "Their orange sale stickers were right on the tags."),
        ("the kettle cost more at the register than on the shelf", "The shelf label and receipt don't match."),
        ("the two for one promotion only took a few cents off", "The sign made it look like the second one was free."),
        ("the new delivery surcharge makes small orders expensive", "The parcel arrived fine but the fee seems steep."),
        ("the price is hard to justify for such a tiny refill", "The larger pack actually works out cheaper."),
        ("the loyalty coupon said ten off but the bill only dropped by five", "Could you check how the discount was applied?"),
        ("the store price jumped overnight for the same chair", "Nothing else about the listing changed."),
        ("the tag showed a member price without mentioning membership", "I only found out when I paid."),
        ("the invoice added a fee that wasn't disclosed in the basket", "It's the extra charge I'm unhappy about."),
        ("the receipt lists two charges for a single scarf", "Please refund the duplicate amount."),
        ("the bundle came to less than buying the pieces separately", "For once the deal was actually worth it."),
    ),
    "delivery_or_stock": (
        ("the courier missed the delivery window again", "I stayed home and nobody turned up."),
        ("the shelf has been empty of refill packs for three weeks", "When will they be restocked?"),
        ("the item was marked available until I got to the pickup desk", "They had none left when I arrived."),
        ("the parcel tracking has not moved since it left the depot", "It was meant to arrive yesterday."),
        ("the website cancelled my order because the size sold out", "It showed plenty available when I paid."),
        ("the courier left my groceries at the wrong building", "The photo isn't of my doorway."),
        ("the store keeps running out of the larger batteries", "The little ones aren't compatible with my device."),
        ("the expedited package took longer than regular shipping", "I paid the upgrade to get it before the weekend."),
        ("half the collection order was removed at pickup", "They said the remaining items hadn't come in."),
        ("the tracking page says delivered but there's nothing at the door", "I've checked the lobby too."),
        ("the restock email came after the item was gone again", "I clicked it straight away."),
        ("the store had the sample model but no boxed units for sale", "The person helping was nice, but I still left empty-handed."),
    ),
    "store_experience": (
        ("the aisles were too narrow to get a stroller through", "Several displays stuck out into the walkway."),
        ("the queue curled around the whole store", "The staff were friendly, but the wait was the main problem."),
        ("the fitting rooms were noticeably dirty", "There was a pile of old tags on the floor."),
        ("the department signs pointed in opposite directions", "I spent ten minutes trying to find cookware."),
        ("the music was so loud that I couldn't hear my friend", "Hard to relax while shopping."),
        ("the entrance ramp was blocked by a stack of baskets", "I had to go around to another door."),
        ("the checkout screens were hard to read in the glare", "A different angle would make a big difference."),
        ("the store was spotless and easy to navigate", "The wide aisles made it a calm trip."),
        ("the air conditioning had the changing room freezing", "I rushed and didn't try everything on."),
        ("the returns line and new purchases line weren't marked", "Everyone kept getting sent to the other queue."),
        ("the shop closed the front door before the posted closing time", "I only needed a quick visit."),
        ("the parking area had no lighting near the side entrance", "It made the evening visit uncomfortable."),
    ),
}
STYLES = (
    lambda event, detail: f"{event.capitalize()}. {detail}",
    lambda event, detail: f"Quick feedback: {event}!",
    lambda event, detail: f"{detail} Main thing: {event}.",
    lambda event, detail: f"idk if anyone else noticed but {event} ??",
)


def build() -> list[dict[str, str]]:
    """Return four phrasings for each scenario under a shared group identifier."""
    return [
        {"comment": style(event, detail), "label": label, "group": f"{label}-{number:02d}"}
        for label, scenarios in SCENARIOS.items()
        for number, (event, detail) in enumerate(scenarios, 1)
        for style in STYLES
    ]


if __name__ == "__main__":
    Path(__file__).with_name("feedback.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in build()), encoding="utf-8"
    )
